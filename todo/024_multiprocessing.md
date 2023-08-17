# Multiprocessing

This would be relevant to `lsp_server` and `llm_server`.

I've tried several times to do this, but have run up against errors where the CUDA context gets set in the main process, and cannot be shared with subprocesses.

I have 2 minimal experiments that are failing as of now, from that error:

```
sandbox/t13_multiprocessing.py
sandbox/t13_multiprocessing_2.py
```

I think the proper solution looks something like this:

1. set_start_method('spawn') on the main process

  - possibly from torch.multiprocessing
  - we might need to patch Thespian to use torch.mp instead of standard python mp

2. instantiate a cuda context on the main process

3. instantiate models on the main process

4. send the models to processes via torch.multiprocessing.Queue

5. There are rules around when you can delete the model off the main process: https://pytorch.org/docs/stable/multiprocessing.html


What's a CUDA context? It's all the state that a process needs to manage and direct GPUs. There are rules over how this gets shared around. I think we're ok if we make `model`s on the main process though, then share them over.
