<p align="center">
  <img width="256" height="256" src="https://raw.githubusercontent.com/freckletonj/uniteai/master/icon.jpeg" alt='uniteai'>
</p>

<p align="center">
<em>UniteAI: Voice-to-text, Local LLM, and GPT, right in your editor.</em>
</p>

---
[![Package version](https://badge.fury.io/py/uniteai.svg)](https://pypi.python.org/pypi/uniteai)

**Requirements:** Python 3

**Editor:** VSCode(ium) or Emacs or any editor with LSP capabilities (most).


## Screencast Demo

[screencast.webm](https://github.com/freckletonj/uniteai/assets/8399149/6cc56405-bf8f-4b1c-89d3-dbe4ff0c794f)

## The Vision

### AIs, Why?

As we integrate more technology into our lives, it's becoming clear that our interactions with these systems will be more and more AI-mediated. This project envisions a future where:

1. **Creation:** We co-create code, books, emails, work outputs, and more with AI.
2. **Management:** AI aids in task and process handling.
3. **Learning:** We learn and explore new concepts with AI.
4. **Entertainment:** Our leisure times are enhanced through AI interaction.

This project hopes build **A Good Interface**.


### But why this project?


* **The Human-AI Team:** feed off each others' strengths

  |                        | **AI** | **You** |
  |:-----------------------|--------|---------|
  | Knows what you want    |        | ✓       |
  | Doesn't hallucinate    |        | ✓       |
  | Knows a ton            | ✓      |         |
  | Thinks super fast      | ✓      |         |
  | Easy to automate tasks | ✓      |         |

* **One-for-All AI Environment:** get your AI Stack in *one* environment, get synergy among the tools.

* **Self-hosted AI Stack:** more control, better security and customization.

* **High speed communication:** Ultimate man-machine flow needs high-speed communication. Symbolic language is best served in a text-editor environment. Natural language integrates seamlessly via voice-to-text.

* **Conclusion:** *Let's get a local AI stack cozy inside a text editor.*


## Quickstart, installing Everything on Ubuntu

You can install more granularly than *everything*, but we'll demo *everything* first.

The only platform-dependent dependency right now is `portaudio`, which I mention in the next section how to install for linux/mac.

1.) Get: `uniteai_lsp`.

```sh
sudo apt install portaudio19-dev
pip install uniteai[all]
uniteai_lsp
```

It will prompt if it should make a default `.uniteai.yml` config for you. Update your preferences, including your OpenAI API key if you want that, and which local language model or transcription models you want.


2.) *Optional:* Then start the longlived LLM server which offers your editor a connection to your local large language model.

```sh
uniteai_llm
```


3.) Install in your editor:

* For **VSCode** get the [`uniteai` extension](https://marketplace.visualstudio.com/publishers/uniteai). Eg in VSCode, `Ctrl-P` then `ext install uniteai.uniteai` .

* For **VSCodium**, VSCode Marketplace files are not compatible, so you'll need to either:

  * Download the prepackaged [`uniteai.vsix`](./clients/vscode/) extension, then:
    ```sh
    codium --install-extension clients/vscode/uniteai.vsix
    ```

  * DIY:
    ```sh
    npm install -g @vscode/vsce
    git clone https://github.com/freckletonj/uniteai
    cd uniteai/clients/vscode
    vsce package
    codium --install-extension uniteai-version.vsix
    ```

* For **Emacs**, copy the [`lsp-mode` config](./clients/emacs/example_lsp_mode_config.el) to your `init.el`.

* For other editors with LSP support (most do), we just need to copy the [emacs/vscode configuration](./clients), and translate it to your editor. Please submit a PR with new editor configs!

## Granular installs

Still refer to the Quickstart section for the main workflow, such as calling `uniteai_lsp` to get your default config made.

Your config determines what modules/features are loaded.

The following makes sure to get your dependencies for each feature.

### Transcription dependencies

```sh
# Debian/Ubuntu
sudo apt install portaudio19-dev  # needed by PyAudio

# Mac
brew install portaudio  # needed by PyAudio

pip install uniteai[transcription]
```

### Local LLM dependencies

```sh
pip install uniteai[local_llm]
```

### OpenAI/ChatGPT dependencies

```sh
pip install uniteai[openai]
```

## Keycombos

Your client configuration determines this, so if you are using the example client config examples in `./clients`:

| VSCode      | Emacs   | Effect                                               |
|:------------|:--------|:-----------------------------------------------------|
| <lightbulb> | M-'     | Show Code Actions Menu                               |
| Ctrl-Alt-g  | C-c l g | Send region to **GPT**, stream output to text buffer |
| Ctrl-Alt-c  | C-c l c | Same, but **ChatGPT**                                |
| Ctrl-Alt-l  | C-c l l | Same, but **Local (eg Falcon) model**                |
| Ctrl-Alt-v  | C-c l v | Start **voice-to-text**                              |
| Ctrl-Alt-s  | C-c l s | Whatevers streaming, stop it                         |


*I'm still figuring out what's most ergonomic, so, I'm accepting feedback.*


## Contributions

### Why?

Because there are **so many cool tools** to yet be added:

* Image creation, eg: *"Write a bulleted plan for a Hero's Journey story about X, and make an image for each scene."*

* Contextualize the AI via reading my emails via POP3, and possibly responding, eg: *"what was that thing my accountant told me not to forget?"*

* Ask my database natural language questions, eg: *"what were my top 10% customers' top 3 favorite products?"*

* Write-ahead for tab-completion, eg: *"Once upon a ____".*

* Chat with a PDF document, eg: *"what do the authors mean by X?"*

* Do some searches, scrape the web, and upload it all into my db.

* Sky's the limit.


### How?

A Key goal of this project is to be **Contributor-Friendly**.

* Make an [Issue](https://github.com/freckletonj/uniteai/issues) with your cool concept, or bug you found.

* [`.todo/`](./todo) is a directory of community "tickets", eg [`.todo/042_my_cool_feature.md`](./todo). Make a ticket or take a ticket, and make a PR with your changes!

* [`./todo/README.md`](./todo/README.md) gives some overview of the library, and advice on building against this library.

* a [`./contrib`](./contrib) directory is where you can add your custom feature. See [`./uniteai/contrib/example.py`](./uniteai/contrib/example.py).

* `.uniteai.yml` configuration chooses which modules to load/not load.

* The code is *well-documented*, *robust*, and *simple*, to reduce friction.

* Adding a feature is as simple as writing some python code, and making use of `uniteai`'s library to directly handle issues like concurrency and communicating/modifying the text editor.


## Misc

### Notes on Local LLMs

The file [`./llm_server.py`](./llm_server.py) launches a TCP server in which the LLM weights are booted up. The `lsp_server` will make calls to this `llm_server`.

The reason is that the `lsp_server` lifecycle is (generally*) managed by the text editor, and LLM models can be really slow to boot up. Especially if you're developing a feature, you do not want the LLM to keep being read into your GPU each time you restart the `lsp_server`.

`*` you don't have to let the editor manage the `lsp_server`. For instance, `eglot` in emacs allows you to launch it yourself, and then the editor client can just bind to the port.


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

## License

Copyright (c) Josh Freckleton. All rights reserved.

Licensed under the [Apache-2.0](https://apache.org/licenses/LICENSE-2.0) license.
