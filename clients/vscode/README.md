
# VSCode Client for UniteAI

# Building

1.
```
cd clients/vscode
npm install
```

2.

# Development

1.
```
cd clients/vscode
npm install
```

2. Open VSCode

3. `Ctrl-Shift-B` starts compiling the client

4. `Ctrl-Shift-D` starts the Run and Debug view

5. `F5` launches the [Extension Development Host](https://code.visualstudio.com/api/get-started/your-first-extension#:~:text=Then%2C%20inside%20the%20editor%2C%20press%20F5.%20This%20will%20compile%20and%20run%20the%20extension%20in%20a%20new%20Extension%20Development%20Host%20window.), and the client should work in here.

# Publishing

0. Refresh your Azure Personal Access Token [link](https://code.visualstudio.com/api/working-with-extensions/publishing-extension#get-a-personal-access-token).

```
vsce login uniteai
# paste token
```

1. bump version in `package.json`

2.

```
npm install -g @vscode/vsce

cd clients/vscode
vsce package  # make shareable .vsix
vsce publish  # put on marketplace
```
