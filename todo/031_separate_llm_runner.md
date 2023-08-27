# 031 LLM Server Should Be Separate Repo

## The LLM Server is useful separately on its own.

## Configurable?

Maybe we should even allow an LSP client to configure whether to launch the LLM Server internally, or separately?

I think though, most clients would start a new LLM Server for each project, so, it would probably load the weights to memory multiple times, and flood your RAM/VRAM. Maybe not possible.

But maybe the LSP client could launch an LLM Server as a background process, if it doesn't already exist?
