# uniteai

Interact with AIs via the editor you already use, right inside your documents.

A key goal is to make this code simple, robust, and contributor-friendly.

Please consider adding to `./contrib`utions, make a PR against the main library, or make an [Issue](https://github.com/freckletonj/uniteai/issues) with your cool concept.


|                        | **AI** | **You** |
|:-----------------------|--------|---------|
| Knows what you want    |        | ✓       |
| Doesn't hallucinate    |        | ✓       |
| Knows a ton            | ✓      |         |
| Thinks super fast      | ✓      |         |
| Easy to automate tasks | ✓      |         |
| Supported in `uniteai` | ✓      | ✓       |


## The Goods

![uniteai screencast](./screencast.gif)


## Capabilities

|                                                |   |
|:-----------------------------------------------|---|
| **Features**                                   |   |
| local voice-to-text                            | ✓ |
| local LLM                                      | ✓ |
| ChatGPT & GPT API                              | ✓ |
| Works via standard LSP                         | ✓ |
| Only enable features you want                  | ✓ |
|                                                |   |
| **Future**                                     |   |
| Document retrieval & Embedding Indexing        | _ |
| Prompt engineering assistants                  | _ |
| Contextualized onyour files, repo, email, etc. | _ |
| Write-ahead for tab-completion                 | _ |
| Contextualized on multiple highlighted regions | _ |
|                                                |   |
| **Editors**                                    |   |
| emacs                                          | ✓ |
| vscode                                         | _ |
| vim                                            | _ |
| jetbrains                                      | _ |
| atom                                           | _ |
|                                                |   |
| **Meta**                                       |   |
| `contrib` dir for community contributions      | ✓ |
| well documented                                | ✓ |
| robust simple code                             | ✓ |


## Setup


### Emacs

#### 0. Setup

```
git clone git@github.com:freckletonj/uniteai
cd uniteai/
pip install -r requirements.txt
sudo apt install portaudio19-dev  # if you want transcription
```


#### 1. Setup emacs's `init.el`

* `lsp-mode`: `lsp-mode` is recommended because it allows multipe LSPs to run in parallel in the same buffer (eg `uniteai` and your python LSP). See [`doc/example_lsp_mode_config.el`](./doc/example_lsp_mode_config.el).

* `EGlot`: See [`doc/example_eglot_config.el`](./doc/example_eglot_config.el):




#### 3. Optional: Run the local LLM server

```
uvicorn llm_server:app --port 8000
```

This reads your `config.yml` (example is in the repo) to find a Transformers-compatible model, eg Falcon, and will run it.

I imagine if you point at the dir of any Transformers-compatible model, this should work.


#### 4. Give it a go.

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


### vscode

Accepting contributions. See [`.doc/`](./doc) for examples in other editors, it's quite simple.


### vim

Accepting contributions. See [`.doc/`](./doc) for examples in other editors, it's quite simple.


### jetbrains

Accepting contributions. See [`.doc/`](./doc) for examples in other editors, it's quite simple.


### atom

Accepting contributions. See [`.doc/`](./doc) for examples in other editors, it's quite simple.


## Misc

### TODO

- [ ] support other editors
- [ ] add cool features


### Falcon LLM Issue:

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
