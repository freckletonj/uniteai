<p align="center">
  <img width="256" height="256" src="https://raw.githubusercontent.com/freckletonj/uniteai/master/icon.jpeg" alt='uniteai'>
</p>

<p align="center">
<em>Your AI Stack in your Editor: Voice-to-text, Local LLM, and GPT, +more.</em>
</p>

---
[![Package version](https://badge.fury.io/py/uniteai.svg)](https://pypi.python.org/pypi/uniteai)  
<a href="https://discord.gg/3K2Q93bug" class="text-decoration-none">
  Join Discord
  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-discord" viewBox="0 0 16 16">
    <path d="M13.545 2.907a13.227 13.227 0 0 0-3.257-1.011.05.05 0 0 0-.052.025c-.141.25-.297.577-.406.833a12.19 12.19 0 0 0-3.658 0 8.258 8.258 0 0 0-.412-.833.051.051 0 0 0-.052-.025c-1.125.194-2.22.534-3.257 1.011a.041.041 0 0 0-.021.018C.356 6.024-.213 9.047.066 12.032c.001.014.01.028.021.037a13.276 13.276 0 0 0 3.995 2.02.05.05 0 0 0 .056-.019c.308-.42.582-.863.818-1.329a.05.05 0 0 0-.01-.059.051.051 0 0 0-.018-.011 8.875 8.875 0 0 1-1.248-.595.05.05 0 0 1-.02-.066.051.051 0 0 1 .015-.019c.084-.063.168-.129.248-.195a.05.05 0 0 1 .051-.007c2.619 1.196 5.454 1.196 8.041 0a.052.052 0 0 1 .053.007c.08.066.164.132.248.195a.051.051 0 0 1-.004.085 8.254 8.254 0 0 1-1.249.594.05.05 0 0 0-.03.03.052.052 0 0 0 .003.041c.24.465.515.909.817 1.329a.05.05 0 0 0 .056.019 13.235 13.235 0 0 0 4.001-2.02.049.049 0 0 0 .021-.037c.334-3.451-.559-6.449-2.366-9.106a.034.034 0 0 0-.02-.019Zm-8.198 7.307c-.789 0-1.438-.724-1.438-1.612 0-.889.637-1.613 1.438-1.613.807 0 1.45.73 1.438 1.613 0 .888-.637 1.612-1.438 1.612Zm5.316 0c-.788 0-1.438-.724-1.438-1.612 0-.889.637-1.613 1.438-1.613.807 0 1.451.73 1.438 1.613 0 .888-.631 1.612-1.438 1.612Z"></path>
  </svg>
  <svg xmlns="http://www.w3.org/2000/svg" height="1em" viewBox="0 0 640 512"><path d="M524.531,69.836a1.5,1.5,0,0,0-.764-.7A485.065,485.065,0,0,0,404.081,32.03a1.816,1.816,0,0,0-1.923.91,337.461,337.461,0,0,0-14.9,30.6,447.848,447.848,0,0,0-134.426,0,309.541,309.541,0,0,0-15.135-30.6,1.89,1.89,0,0,0-1.924-.91A483.689,483.689,0,0,0,116.085,69.137a1.712,1.712,0,0,0-.788.676C39.068,183.651,18.186,294.69,28.43,404.354a2.016,2.016,0,0,0,.765,1.375A487.666,487.666,0,0,0,176.02,479.918a1.9,1.9,0,0,0,2.063-.676A348.2,348.2,0,0,0,208.12,430.4a1.86,1.86,0,0,0-1.019-2.588,321.173,321.173,0,0,1-45.868-21.853,1.885,1.885,0,0,1-.185-3.126c3.082-2.309,6.166-4.711,9.109-7.137a1.819,1.819,0,0,1,1.9-.256c96.229,43.917,200.41,43.917,295.5,0a1.812,1.812,0,0,1,1.924.233c2.944,2.426,6.027,4.851,9.132,7.16a1.884,1.884,0,0,1-.162,3.126,301.407,301.407,0,0,1-45.89,21.83,1.875,1.875,0,0,0-1,2.611,391.055,391.055,0,0,0,30.014,48.815,1.864,1.864,0,0,0,2.063.7A486.048,486.048,0,0,0,610.7,405.729a1.882,1.882,0,0,0,.765-1.352C623.729,277.594,590.933,167.465,524.531,69.836ZM222.491,337.58c-28.972,0-52.844-26.587-52.844-59.239S193.056,219.1,222.491,219.1c29.665,0,53.306,26.82,52.843,59.239C275.334,310.993,251.924,337.58,222.491,337.58Zm195.38,0c-28.971,0-52.843-26.587-52.843-59.239S388.437,219.1,417.871,219.1c29.667,0,53.307,26.82,52.844,59.239C470.715,310.993,447.538,337.58,417.871,337.58Z"/></svg>
</a>
  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-discord" viewBox="0 0 16 16">
    <path d="M13.545 2.907a13.227 13.227 0 0 0-3.257-1.011.05.05 0 0 0-.052.025c-.141.25-.297.577-.406.833a12.19 12.19 0 0 0-3.658 0 8.258 8.258 0 0 0-.412-.833.051.051 0 0 0-.052-.025c-1.125.194-2.22.534-3.257 1.011a.041.041 0 0 0-.021.018C.356 6.024-.213 9.047.066 12.032c.001.014.01.028.021.037a13.276 13.276 0 0 0 3.995 2.02.05.05 0 0 0 .056-.019c.308-.42.582-.863.818-1.329a.05.05 0 0 0-.01-.059.051.051 0 0 0-.018-.011 8.875 8.875 0 0 1-1.248-.595.05.05 0 0 1-.02-.066.051.051 0 0 1 .015-.019c.084-.063.168-.129.248-.195a.05.05 0 0 1 .051-.007c2.619 1.196 5.454 1.196 8.041 0a.052.052 0 0 1 .053.007c.08.066.164.132.248.195a.051.051 0 0 1-.004.085 8.254 8.254 0 0 1-1.249.594.05.05 0 0 0-.03.03.052.052 0 0 0 .003.041c.24.465.515.909.817 1.329a.05.05 0 0 0 .056.019 13.235 13.235 0 0 0 4.001-2.02.049.049 0 0 0 .021-.037c.334-3.451-.559-6.449-2.366-9.106a.034.034 0 0 0-.02-.019Zm-8.198 7.307c-.789 0-1.438-.724-1.438-1.612 0-.889.637-1.613 1.438-1.613.807 0 1.45.73 1.438 1.613 0 .888-.637 1.612-1.438 1.612Zm5.316 0c-.788 0-1.438-.724-1.438-1.612 0-.889.637-1.613 1.438-1.613.807 0 1.451.73 1.438 1.613 0 .888-.631 1.612-1.438 1.612Z"></path>
  </svg>
  <svg xmlns="http://www.w3.org/2000/svg" height="1em" viewBox="0 0 640 512"><path d="M524.531,69.836a1.5,1.5,0,0,0-.764-.7A485.065,485.065,0,0,0,404.081,32.03a1.816,1.816,0,0,0-1.923.91,337.461,337.461,0,0,0-14.9,30.6,447.848,447.848,0,0,0-134.426,0,309.541,309.541,0,0,0-15.135-30.6,1.89,1.89,0,0,0-1.924-.91A483.689,483.689,0,0,0,116.085,69.137a1.712,1.712,0,0,0-.788.676C39.068,183.651,18.186,294.69,28.43,404.354a2.016,2.016,0,0,0,.765,1.375A487.666,487.666,0,0,0,176.02,479.918a1.9,1.9,0,0,0,2.063-.676A348.2,348.2,0,0,0,208.12,430.4a1.86,1.86,0,0,0-1.019-2.588,321.173,321.173,0,0,1-45.868-21.853,1.885,1.885,0,0,1-.185-3.126c3.082-2.309,6.166-4.711,9.109-7.137a1.819,1.819,0,0,1,1.9-.256c96.229,43.917,200.41,43.917,295.5,0a1.812,1.812,0,0,1,1.924.233c2.944,2.426,6.027,4.851,9.132,7.16a1.884,1.884,0,0,1-.162,3.126,301.407,301.407,0,0,1-45.89,21.83,1.875,1.875,0,0,0-1,2.611,391.055,391.055,0,0,0,30.014,48.815,1.864,1.864,0,0,0,2.063.7A486.048,486.048,0,0,0,610.7,405.729a1.882,1.882,0,0,0,.765-1.352C623.729,277.594,590.933,167.465,524.531,69.836ZM222.491,337.58c-28.972,0-52.844-26.587-52.844-59.239S193.056,219.1,222.491,219.1c29.665,0,53.306,26.82,52.843,59.239C275.334,310.993,251.924,337.58,222.491,337.58Zm195.38,0c-28.971,0-52.843-26.587-52.843-59.239S388.437,219.1,417.871,219.1c29.667,0,53.307,26.82,52.844,59.239C470.715,310.993,447.538,337.58,417.871,337.58Z"/></svg>
**Requirements:** Python 3

**Editor:** VSCode(ium) or Emacs or Any Editor with LSP capabilities (most).

## The Mission
<br>
<p align="center">
<em>The future is about Humans Augmented with AIs.</em>
</p>
<br>
<br>

We need our **AI Stack** (Online, or local models)

Inside a **convenient interface** (Text Editors > Web UIs)

Friendly with **any editor** (The project is an [LSP](https://en.wikipedia.org/wiki/Language_Server_Protocol) and therefore highly portable)

And **close to the code** (It's easy to tweak and add features. All the logic happens in friendly *python* code, not bespoke one-off editor code).


## Screencast Demo

### Some Core Features

[screencast.webm](https://github.com/freckletonj/uniteai/assets/8399149/6cc56405-bf8f-4b1c-89d3-dbe4ff0c794f)

### Document Chat (***NEW***)

[screencast_document_chat.webm](https://github.com/freckletonj/uniteai/assets/8399149/b20eea79-431e-44bb-b782-24c57edc1b88)


## Quickstart, installing Everything

You can install more granularly than *everything*, but we'll demo *everything* first.

1.) Make sure Python 3 + Pip is installed.

```sh
python --version
pip --version

# or

python3 --version
pip3 --version
```


2.) The only platform-dependent dependency right now is `portaudio`, and that is only needed if you want speech-to-text/transcription.

```sh
# Mac
brew install portaudio

# Ubuntu/Debian
sudo apt install portaudio19-dev
```

3.) Get: `uniteai_lsp`, build a config.

```sh
pip3 install --user "uniteai[all]" # install deps for all features
uniteai_lsp                        # on mac, this may only appear if you open a new terminal
cat .uniteai.yml                   # checkout the config

# if you want global config (unnecessary, but you probably do,
# otherwise it just searches your current dir):
mv .uniteai.yml ~/
```

It will prompt if it should make a default `.uniteai.yml` config for you. Update your preferences, including your OpenAI API key if you want that, and which local language model or transcription models you want.


4.) *Optional:* Then start the longlived LLM server which offers your editor a connection to your local large language model.

```sh
uniteai_llm
```


5.) Install in your editor:

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

If you did `pip install "uniteai[all]"`, ignore this section!

Still refer to the Quickstart section for the main workflow, such as calling `uniteai_lsp` to get your default config made.

Your config determines what modules/features are loaded.

The following makes sure to get your dependencies for each feature. This will become more relevant when more community features are added.

### Transcription dependencies

```sh
# Debian/Ubuntu
sudo apt install portaudio19-dev  # needed by PyAudio

# Mac
brew install portaudio  # needed by PyAudio

pip3 install "uniteai[transcription]"
```

### Local LLM dependencies

```sh
pip3 install "uniteai[local_llm]"
```

### OpenAI/ChatGPT dependencies

```sh
pip3 install "uniteai[openai]"
```

## Keycombos

Your client configuration determines this, so if you are using the example client config examples in `./clients`:

| VSCode      | Emacs   | Effect                                               |
|:------------|:--------|:-----------------------------------------------------|
| <lightbulb> | M-'     | Show Code Actions Menu                               |
| Ctrl-Alt-d  | C-c l d | Do semantic search on a document                     |
| Ctrl-Alt-g  | C-c l g | Send region to **GPT**, stream output to text buffer |
| Ctrl-Alt-c  | C-c l c | Same, but **ChatGPT**                                |
| Ctrl-Alt-l  | C-c l l | Same, but **Local (eg Falcon) model**                |
| Ctrl-Alt-v  | C-c l v | Start **voice-to-text**                              |
| Ctrl-Alt-s  | C-c l s | Whatevers streaming, stop it                         |


* *I'm still figuring out what's most ergonomic, so, I'm accepting feedback.*

* `Ctrl-Alt-d` on ubuntu means defaults to "minimize all windows". You can [disable](https://askubuntu.com/a/177994/605552) that.


# "Neural" Document Lookup

For the `document` feature, you can reference one of multiple document types, and lookup passages with a similar "gist" to them (semantic similarity search).

Check that your `.uniteai.yaml` config has `uniteai.document` enabled.

You can use links to: YouTube (will read transcripts), Arxiv papers, PDFs, Git repos, or any HTML.

To use this feature, write some YAML, highlight it, and hit `C-c l d` (emacs) or `C-A-d` (vscode).

```yaml
query:
docs:
  - title: (optional)
    url: ...
  - title: ...
    url: ...
```

It will take a couple minutes for long documents to get an embedding for each chunk it finds in the document, but that then gets cached and goes fast afterward.

[More details.](./uniteai/document/README.md)


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
