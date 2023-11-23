'''

Play around with getting ctransformers working

'''

from ctransformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import TextIteratorStreamer
import threading
import torch
from transformers.generation import StoppingCriteriaList

path = './zephyr-7b-beta.Q8_0.gguf'
model = AutoModelForCausalLM.from_pretrained(path, hf=True)
tokenizer = AutoTokenizer.from_pretrained('HuggingFaceH4/zephyr-7b-beta')  # NOTE: HF tokenizer
a = model.generate(tokenizer('hi', return_tensors='pt').input_ids)
b = tokenizer.decode(a[0])
print(b)
