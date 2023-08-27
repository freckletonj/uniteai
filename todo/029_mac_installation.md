# 029 Mac Installation Improvements

Based on helping a non-dev install this, there are a lot of expectations I had of how the environment would be set up, and it isn't, and therefore installation is harder than need be.

```
brew needs to be installed

pip location needs to be on the PATH

portaudio libs need to be on the LIBRARY path

log_file.log Read only file system error

SSL CERTIFICATE_VERIFY_FAILED, self signed certificate in cert chain
  solution, run this (why?):
  /Applications/Python 3.10/Install\ Certificates.command

Command-option-D is a system level thing, so doesn't work as a keycombo

Server times out when downloading document stuff

Document stuff outputs on stdout/stderr, breaking things

Signify when the server is warmed up
```
