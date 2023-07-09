# 002 Newline Management :DONE:

When writing in parallel with LSP, if you make newlines above the LLM's additions, its pointer does not get updated, and will stream its response all scattershot.

Perhaps we could record on the LSP server everytime a newline is added/removed (by the user or by yet another concurrent LLM) in order to move their pointer.


# Notes

This was ultimately solved by giving the LSP a tagged block it could write within, EG:

```
You can type before

:START_TRANSCRIPTION:
Here is the stuff you're saying.
:END_TRANSCRIPTION:

Or after, and it's all good.
```
