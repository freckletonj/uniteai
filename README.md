# uniteai

Interact with an AI right in your text document.

|                        | **AI** | **You** |
|:-----------------------|--------|---------|
| Knows what you want    |        | ✓       |
| Doesn't hallucinate    |        | ✓       |
| Knows a ton            | ✓      |         |
| Thinks super fast      | ✓      |         |
| Easy to automate tasks | ✓      |         |
| Supported in `uniteai` | ✓      | ✓       |


Features

|                                           |   |
|:------------------------------------------|---|
| local voice-to-text                       |   |
| local LLM                                 |   |
| ChatGPT & GPT                             |   |
|                                           |   |
| **Editors**                               |   |
| emacs                                     |   |
| vscode                                    | _ |
| vim                                       | _ |
|                                           |   |
| **Meta**                                  |   |
| well documented                           |   |
| robust simple code                        |   |
| `contrib` dir for community contributions |   |
|                                           |   |

A hackable implementation that lets you roll-your-own prompt-engineering/custom models/plug into other tools etc.

It is tested with Emacs, but the LSP Server should work on most editors. More editors to come!


## The Goods

![uniteai screencast](./screencast.gif)

Right now this is set up to work automatically on `.llm` files only, because Eglot cannot run multiple LSPs in parallel. So if you use a python LSP, this won't work in conjunction. This will be an easy fix eventually.


## Setup

This currently supports Emacs (but I would love to add support for more editors).

So, the steps to run:

### 0. Setup

```
pip install -r requirements.txt
sudo apt install portaudio19-dev
```


### 1. Setup emacs's `init.el`

* `lsp-mode`: `lsp-mode` is recommended because it allows multipe LSPs to run in parallel in the same buffer (eg `uniteai` and your python LSP). See [`doc/example_lsp_mode_config.el`](./doc/example_lsp_mode_config.el).

* `EGlot`: See [`doc/example_eglot_config.el`](./doc/example_eglot_config.el):


### 2. Run the LSP Server

```
pip install -r requirements.txt

python lsp_server.py
```


### 3. Optional: Run the local LLM server

```
uvicorn llm_server:app --port 8000
```

This reads your `config.yml` (example is in the repo) to find a Transformers-compatible model, eg Falcon, and will run it.

I imagine if you point at the dir of any Transformers-compatible model, this should work.


### 4. Give it a go.

**Keycombos**

Your client configuration determines this, so if you are using the example client config examples in `./doc`:

| Keycombo | Effect                                           |
|:---------|:-------------------------------------------------|
| M-'      | Show Code Actions Menu                           |
|          |                                                  |
| C-c l g  | Send region to GPT, stream output to text buffer |
| C-c l c  | Same, but ChatGPT                                |
| C-c l l  | Same, but local (eg Falcon) model                |
| C-c C-c  | Same as `C-c l l` but quicker to hit             |
|          |                                                  |
| C-c l s  | Whatevers streaming, stop it                     |


## Misc

### TODO

- [ ] support VSCode and other editors
- [ ] Add ability to work in any buffer with any LSP. Eglot can't handle multi LSPs, but LSP-mode should be able. The issue with LSP-mode is I don't think it supports working over TCP, so reloading the server in dev env is annoying.
- [ ] How should we package this up more nicely?


### Falcon Issue:

If Falcon runs on multiple threads, its cache has an issue. You need a separate `modelling_RW.py` that makes sure it never tries to cache.
https://github.com/h2oai/h2ogpt/pull/297

Replacing `cos_sim` with this seems to do the trick:

```python
def cos_sin(
    self,
    seq_len: int,
    device="cuda",
    dtype=torch.bfloat16,
) -> torch.Tensor:
    t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
    freqs = torch.einsum("i,j->ij", t, self.inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1).to(device)

    if dtype in [torch.float16, torch.bfloat16]:
        emb = emb.float()

    cos_cached = emb.cos()[None, :, :]
    sin_cached = emb.sin()[None, :, :]

    cos_cached = cos_cached.type(dtype)
    sin_cached = sin_cached.type(dtype)

    return cos_cached, sin_cached
```

A separate bitsandbytes issue remains unresolved, but is less serious than the above.
https://github.com/h2oai/h2ogpt/issues/104
https://github.com/TimDettmers/bitsandbytes/issues/162
