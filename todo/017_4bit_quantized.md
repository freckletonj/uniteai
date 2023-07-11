# 017 Support 4bit Quantized Models

AutoForCausalLM claims to support `load_in_4bit`. In testing it did not work with any 4bit models, eg MosaicML Storywriter.

We could make an alternative loader instead of `.from_pretrained` and gain inspiration from how oobabooga is doing it, or 4bit llama inference. I propose a new module `quantize.py` with a loader function in it. Then, when the llm server loads the model, it could potentially use this loader if `4bit` was found in the model name.

Ex: https://github.com/oobabooga/text-generation-webui/blob/a81cdd1367c3a4cf77e37fc0ab21ba7990bba843/modules/GPTQ_loader.py#L139
