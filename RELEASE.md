# Release Notes

## Server

1. update version in toml file

2. Tag it

```
git tag -a v0.0.0 -m "Release version 0.0.0"
git push origin v0.0.0
```

3. Update `CHANGES.md`

4. `make publish_pypi`


## VSCode Client


1. update version in `package.json`

2. Publish it
```
cd clients/vscode
vsce package  # make shareable .vsix
vsce publish  # put on marketplace
```

3. You may want to remove older vsix binaries


## Emacs

get this sucker on MELPA!
