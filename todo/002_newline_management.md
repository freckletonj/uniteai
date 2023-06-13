# 002 Newline Management

When writing in parallel with LSP, if you make newlines above the LLM's additions, its pointer does not get updated, and will stream its response all scattershot.

Perhaps we could record on the LSP server everytime a newline is added/removed (by the user or by yet another concurrent LLM) in order to move their pointer.
