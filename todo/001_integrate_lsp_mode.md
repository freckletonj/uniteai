# 001 Integrate `lsp-mode`

This will allow multiple LSPs to operate on the same buffer (eg LLM mode + Python mode)


## Notes

- Comment from @Lenbok

```
lsp-mode definitely does support TCP connections, have a look in the lsp-clients folder for examples (the openscad, erlang, and php ones all make use of lsp-tcp-connection function in their setup).
```

- Comment from @yyoncho

```
You can use lsp-tcp-connection or what lsp-gdscript.el is doing if the server is not supposed to be started by Emacs.
```
