# bumps-webview-client

This is the Javascript + HTML client for the webview interface in Bumps

## Starting the server

```sh
python -m bumps.webview.server
```

## Recommended IDE Setup

[VSCode](https://code.visualstudio.com/) + [Volar](https://marketplace.visualstudio.com/items?itemName=Vue.volar) (and disable Vetur) + [TypeScript Vue Plugin (Volar)](https://marketplace.visualstudio.com/items?itemName=Vue.vscode-typescript-vue-plugin).

## Customize configuration

See [Vite Configuration Reference](https://vitejs.dev/config/).

## Project Setup

```sh
npm install
```

### Compile and Hot-Reload for Development

```sh
npm run dev
```

### Compile and Minify
This will create a new folder `./dist/<client version>/` and build the client there,
with a unique tag in the name (suitable for local testing), e.g. `./dist/<client version>/assets/index.BaQLi6ia.js`

```sh
npm run build
```

### Compile for publishing to NPM
This will create a new folder `./dist/<client version>/` and build the client
with reproducible artifact names, e.g. `./dist/<client version>/assets/index.js`
(suitable for packaging and publishing to NPM)

```sh
npm run build_prod
```

# Publishing new client versions:
(...after checking to make sure there aren't extraneous files in this folder)
```sh
npm version patch
npm publish
```

and then
```sh
git commit package.json -m "webview client version bump"
git pull
git push
```
