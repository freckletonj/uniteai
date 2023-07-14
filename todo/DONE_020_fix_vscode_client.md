# 020: Fix VSCode Client

I'm not a VSCoder, and barely got the extension over the hurdle of working. But it's ugly.

I [copied a vscode example](https://github.com/microsoft/vscode-extension-samples/tree/main/lsp-sample), so there are some things I want to understand better and clean up:

* There are 2 `package.json`s. Why? In the [original](https://github.com/microsoft/vscode-extension-samples/tree/main/lsp-sample), there's an outer `package.json`, then another for each of a server and client. It seems like a client should be paired down to just one.

* The example also had a lot of testing stuff. I havent' committed that stuff, but I'm sure I've left dangling references. Better than cleaning them up would be to actually implement some tests for the client.

* Etc. I'm sure I messed up much more.
