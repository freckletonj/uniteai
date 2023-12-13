'''

Launch an LLM locally, and serve it.

----------
RUN:
    uniteai_llm
      # or
    uvicorn uniteai.llm_server:app --port 8000

----------

NOTE: Multiprocessing could potentially be nice instead of Threads,
     BUT Threads is working great, AND:

I've Struggled to get MP to work.

You need to place at top of file:
    mp.set_start_method('spawn')
CUDA requires this, but Uvicorn already forces it to be `fork`.

Perhaps Uvicorn has some nice spawn-friendly process ability?

If you can get that to work, swap for the Thread lines in the llm_streaming
endpoint:
    q = mp.Manager().Queue()
    mp.Process(
        target=local_llm_stream_,
        args=(
            request,
            q,
            local_llm_stop_event
        )).start()

'''


from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel
from transformers.generation import StoppingCriteriaList
from typing import List
import threading
import torch
from uniteai.common import get_nested
from uniteai.config import load_config
import uvicorn
from transformers import TextIteratorStreamer
import llama_cpp
import queue
import time


device = 'cuda' if torch.cuda.is_available() else 'cpu'  # note: this isn't relevant to GGUF models


##################################################
# Load Model

def load_model(args):
    name_or_path = args['model_name_or_path']
    print(f'Loading model: {name_or_path}')

    # T5
    if 't5' in name_or_path:
        from transformers import (
            T5Tokenizer,
            T5ForConditionalGeneration,
        )
        tokenizer = T5Tokenizer.from_pretrained(name_or_path)
        model = T5ForConditionalGeneration.from_pretrained(name_or_path, device_map='auto')
        return tokenizer, model

    # Llama-CPP-Python: GGUF, GPTQ, AWQ
    elif ('gguf' in name_or_path.lower() or
          'gptq' in name_or_path.lower() or
          'awq' in name_or_path.lower()):
        from llama_cpp import Llama
        model = Llama(
            model_path=name_or_path,
            verbose=False
        )
        tokenizer = None
        return tokenizer, model

    # Transformers (should support many models)
    else:
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            name_or_path,
        )
        revision = {'revision': args['model_commit']} if 'model_commit' in args else {}
        device_map = {'device_map': args['device_map']} if 'device_map' in args else {}
        load_in_8bit = {'load_in_8bit': args['load_in_8bit']} if 'load_in_8bit' in args else {}
        load_in_4bit = {'load_in_4bit': args['load_in_4bit']} if 'load_in_4bit' in args else {}
        trust_remote_code = {'trust_remote_code': args['trust_remote_code']} if 'trust_remote_code' in args else {}

        if 'torch_dtype' in args:
            ty_arg = args['torch_dtype']
            if ty_arg in {'float16', 'torch.float16'}:
                torch_dtype = {'torch_dtype': torch.float16}
            elif ty_arg in {'bf16', 'torch.bf16, bfloat16, torch.bfloat16'}:
                torch_dtype = {'torch_dtype': torch.bfloat16}
            else:
                torch_dtype = {}


        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=name_or_path,
            **trust_remote_code,  # needed by eg Falcon, and MosaicML
            **revision,
            **device_map,
            **load_in_8bit,
            **load_in_4bit,
            **torch_dtype,
        )
        return tokenizer, model


##################################################
# Initialization

config = load_config()
args = config['local_llm']


##################################################
# Types

class AutocompleteRequest(BaseModel):
    text: str
    max_length: int = 200
    do_sample: bool = True
    top_k: int = 10
    num_return_sequences: int = 1


class AutocompleteResponse(BaseModel):
    generated_text: str


app = FastAPI()

@app.on_event("startup")
def initialize_model():
    global tokenizer, model
    tokenizer, model = load_model(args)


##################################################
# Local LLM

# for early termination of streaming
local_llm_stop_event = threading.Event()

# when streaming, a chunk must end in newline
STREAM_TOK = '\n'


####################
# llama-cpp-python

class QueueIterator:
    ''' Allow a separate thread to fill a queue, and turn it into a nice
    iterator to behave like transformer's TextIteratorStreamer. This queue can
    be stopped by `put`ting a `None` on it.'''

    def __init__(self):
        self.queue = queue.Queue()

    def __iter__(self):
        return self

    def put(self, text):
        self.queue.put(text)

    def __next__(self):
        item = self.queue.get()

        # Stop signal received
        if item is None:
            raise StopIteration

        return item


def llama_cpp_stream_(request, streamer, local_llm_stop_event):
    def custom_stopping_criteria(stop_event):
        def f(input_ids: torch.LongTensor,
              score: torch.FloatTensor,
              **kwargs) -> bool:
            global local_llm_stop_event
            return stop_event.is_set()
        return f
    stopping_criteria = llama_cpp.StoppingCriteriaList([
        custom_stopping_criteria(local_llm_stop_event)
    ])

    stream = model(
        request.text,
        max_tokens=128,
        stream=True,
        echo=False,  # echo the prompt back as output
        stopping_criteria=stopping_criteria,
    )

    for output in stream:
        if output['choices'][0]['finish_reason'] == 'stop':
            streamer.put(None)
        else:
            streamer.put(output['choices'][0]['text'])


####################
# Transformers

def transformer_stream_(request, streamer, local_llm_stop_event):
    def custom_stopping_criteria(stop_event):
        def f(input_ids: torch.LongTensor,
              score: torch.FloatTensor,
              **kwargs) -> bool:
            return stop_event.is_set()
        return f
    stopping_criteria = StoppingCriteriaList([
        custom_stopping_criteria(local_llm_stop_event)
    ])

    toks = tokenizer([request.text], return_tensors='pt')
    input_ids = toks.input_ids.to(device)
    attention_mask = toks.attention_mask.to(device)

    generation_kwargs = dict(
        inputs=input_ids,
        attention_mask=attention_mask,
        streamer=streamer,
        max_length=request.max_length,
        do_sample=request.do_sample,
        top_k=request.top_k,
        num_return_sequences=request.num_return_sequences,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,  # for open-end generation
        stopping_criteria=stopping_criteria,
        # important for when model is accessed from other threads
        #   https://github.com/h2oai/h2ogpt/pull/297/files
        use_cache=False,
    )
    try:
        model.generate(**generation_kwargs)  # blocks
    except RuntimeError as e:
        streamer.on_finalized_text(f'\n<LLM SERVER ERROR: {e}>', stream_end=True)
    print('DONE GENERATING')


####################
# Choose model setup based on model type

def stream_model_setup(model):
    '''The model is globally available.'''
    if isinstance(model, llama_cpp.llama.Llama):
        streamer = QueueIterator()
        return streamer, llama_cpp_stream_

    else:
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_special_tokens=True,  # eg <|endoftext|>
            skip_prompt=True,  # don't return prompt
        )
        return streamer, transformer_stream_


def stream_results(streamer, stop_event):
    for x in streamer:
        # Stream response, and stop early if requested
        if not stop_event.is_set():
            print(f'yield: {x}')
            yield (
                AutocompleteResponse(generated_text=x).json() + STREAM_TOK
            ).encode('utf-8')
        else:
            # print('STOP WAS SET')
            break


##################################################
#

@app.post("/local_llm_stream", response_model=List[AutocompleteResponse])
def local_llm_stream(request: AutocompleteRequest):
    ''' Stream the response as the model generates it. '''
    global local_llm_stop_event

    local_llm_stop_event.clear()
    streamer, stream_fn = stream_model_setup(model)

    threading.Thread(target=stream_fn,
                     args=(
                         request,
                         streamer,
                         local_llm_stop_event
                     )).start()

    try:
        return StreamingResponse(
            stream_results(streamer, local_llm_stop_event),
            media_type="application/json"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/local_llm_stream_stop")
def local_llm_stream_stop():
    ''' End ongoing inferrence '''
    global local_llm_stop_event
    local_llm_stop_event.set()
    return Response(status_code=200)


def main():
    uvicorn.run("uniteai.llm_server:app",
                host=get_nested(config, ['local_llm', 'host']),
                port=get_nested(config, ['local_llm', 'port']))


if __name__ == "__main__":
    main()
