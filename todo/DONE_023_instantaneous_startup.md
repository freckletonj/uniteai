# 023 Instantaneous Startup

Features should be instantiated after the LSP successfully connects, that should be enough to get an instantaneous startup.

Done, by loading modules after the `INITIALIZED` signal is received by the server.
