'''

Launch an LLM locally, and serve it.

----------
RUN:
    uvicorn llm_server:app --port 8000


----------
NOTE: Multiprocessing may be nice instead of Threads, BUT:

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
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer
)
from transformers.generation import StoppingCriteriaList
from typing import List
import threading
import torch
import yaml
import multiprocessing as mp
import logging


##################################################
# Initialization

with open(".uniteai.yml", 'r') as file:
    config = yaml.safe_load(file)
    model_path = config['model_path']
    model_commit = config['model_commit']  # doesn't work yet in `transformers`


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
def load_model():
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
    )
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        revision='DOESNT ACTUALLY WORK',
        device_map="auto",
        load_in_8bit=True,
    )


##################################################
# Local LLM

# for early termination of streaming
local_llm_stop_event = mp.Event()

# when streaming, a chunk must end in newline
STREAM_TOK = '\n'


def local_llm_stream_(request, streamer, local_llm_stop_event):
    def custom_stopping_criteria(local_llm_stop_event):
        def f(input_ids: torch.LongTensor,
              score: torch.FloatTensor,
              **kwargs) -> bool:
            return local_llm_stop_event.is_set()
        return f
    stopping_criteria = StoppingCriteriaList([
        custom_stopping_criteria(local_llm_stop_event)
    ])

    toks = tokenizer([request.text], return_tensors='pt').to('cuda')
    generation_kwargs = dict(
        inputs=toks.input_ids.cuda(),
        attention_mask=toks.attention_mask,
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
        # NOTE: Falcon randomly produces errors like:
        #       mat1 and mat2 shapes cannot be multiplied
        streamer.on_finalized_text(f'\n<LLM SERVER ERROR: {e}>',
                                   stream_end=True)
    print('DONE GENERATING')


def stream_results(streamer):
    for x in streamer:
        # Stream response, and stop early if requested
        if not local_llm_stop_event.is_set():
            print(f'yield: {x}')
            yield (
                AutocompleteResponse(generated_text=x).json() + STREAM_TOK
            ).encode('utf-8')
        else:
            print('STOP WAS SET')
            break


@app.post("/local_llm_stream", response_model=List[AutocompleteResponse])
def local_llm_stream(request: AutocompleteRequest):
    ''' Stream the response as the model generates it. '''
    local_llm_stop_event.clear()
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_special_tokens=True  # eg <|endoftext|>
    )
    threading.Thread(target=local_llm_stream_,
                     args=(
                         request,
                         streamer,
                         local_llm_stop_event
                     )).start()

    try:
        return StreamingResponse(
            stream_results(streamer),
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
