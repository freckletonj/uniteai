# 025 Friendly Config Construction

Right now, it's a bit cumbersome. The LSP should be able to:

* load even without any config found
* if no config is found, it should create one in the local directory, and notify the user (what LSP feature?)
* it should notify the user of the option to place config in global home dir.
* it should also always display a notification of where the config is loading from, but user can toggle that off via config
