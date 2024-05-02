// import { loadPyodideAndPackages } from './pyodide_worker.mjs';
import { expose } from 'comlink';
import { loadPyodide } from 'pyodide';
import type { PyodideInterface } from 'pyodide';
const DEBUG = true;

// this is provided by vite.config.js as a "define" variable
declare const BUMPS_WHEEL_FILE: string;

var pyodide: PyodideInterface;
// import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v0.23.2/full/pyodide.mjs";

async function loadPyodideAndPackages() { // loads pyodide
    pyodide = await loadPyodide({
        indexURL: "https://cdn.jsdelivr.net/pyodide/v0.25.1/full/",

    }); // run the function and wait for the result (base library)

    await pyodide.loadPackage(["matplotlib", "numpy", "scipy", "pytz", "h5py", "micropip"]); // waits until these python packpages are loaded to continue
  
    //import reductus library with micropip
    let api = await pyodide.runPythonAsync(`
    import micropip
    await micropip.install("./wheels/${BUMPS_WHEEL_FILE}")
    await micropip.install("plotly")
    await micropip.install("mpld3")
    await micropip.install("blinker")
    await micropip.install("dill")

    from bumps.webview.server import api

    wrapped_api = {}
  
    def expose(method, method_name):
        def wrapper(args):
            # print("args:", args)
            pyargs = args.to_py() if args is not None else []
            #print(method_name, "pyargs:", pyargs)
            result = method(*pyargs)
            print("result of", method_name, str(result))
            return result
  
        return wrapper
  
    for method_name, method in api.REGISTRY.items():
        print("wrapping:", method_name)
        wrapped = expose(method, method_name)
        wrapped_api[method_name] = wrapped
  
    wrapped_api
    `);
    return api;
}

// export { loadPyodideAndPackages };

let pyodideReadyPromise = loadPyodideAndPackages(); // run the functions stored in lines 4

type EventCallback = (message?: any) => any;

export class Server {
    handlers: { [signal: string]: EventCallback[] }
    nativefs: any

    constructor() {
        this.handlers = {};
        this.nativefs = null;
        this.init();
    }

    async init() {
        const api = await pyodideReadyPromise;
        const defineEmit = await pyodide.runPythonAsync(`
            def defineEmit(server):
                api.emit = server.asyncEmit;
            
            defineEmit
         `);
        await defineEmit(this);
    }

    async addHandler(signal: string, handler: EventCallback) {
        const signal_handlers = this.handlers[signal] ?? [];
        signal_handlers.push(handler);
        if (DEBUG) {
            console.log(`adding handler: ${signal}`);
        }
        if (signal === 'connect') {
            await pyodideReadyPromise;
            await handler();
        }
        this.handlers[signal] = signal_handlers;
    }

    async removeHandler(signal: string, handler: EventCallback) {
        let signal_handlers = this.handlers[signal] ?? [];
        signal_handlers = signal_handlers.filter((h) => {
            if (h === handler) {
                console.log('matching worker handler found, removing: ', handler);
                return false;
            }
            return true;
        })
        this.handlers[signal] = signal_handlers;
    }

    async mount(dirHandle: FileSystemDirectoryHandle) {
        // const dirHandle = await self.showDirectoryPicker();
        console.log({dirHandle});   
        const nativefs = await pyodide.mountNativeFS("/home/pyodide/user_mount", dirHandle);
        this.nativefs = nativefs;
    }

    async syncFS() {
        let r = await this.nativefs?.syncfs?.();
    }
    
    async asyncEmit(signal: string, message?: any) {
        const jsMessage = message?.toJs({dict_converter: Object.fromEntries}) ?? null;
        console.log('server emit:', signal, jsMessage);
        const handlers = this.handlers[signal] ?? [];
        for (let handler of handlers) {
            handler(jsMessage);
        }
    }

    async onAsyncEmit(signal: string, ...args: any[]) {
        console.log('emit!', signal); // , args);
        const api = await pyodideReadyPromise;
        const callback = (args[args.length - 1] instanceof Function) ? args.pop() : null;
        const result = await api.get(signal)(args);
        const jsResult = result?.toJs({dict_converter: Object.fromEntries}) ?? null;
        if (callback !== null) {
            await callback(jsResult);
        }
        return jsResult;
    }

}

expose(Server);
