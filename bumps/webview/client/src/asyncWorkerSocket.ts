import { wrap, proxy } from 'comlink';
import type { Endpoint, Remote } from 'comlink';
// import './standalone_worker';
import type { Server } from './standalone_worker';
import { off } from 'process';

export function io() {
  const worker = new Worker(new URL("./standalone_worker.ts", import.meta.url), {type: 'module'});
  const socket = new AsyncSocket(worker);
  return socket;
};

type EventCallback = (message: any) => any;

export class AsyncSocket {
  proxy_lookups: Map<EventCallback, typeof proxy<EventCallback>>;
  handlers: { [signal: string]: EventCallback[] }
  ServerPromise: Promise<Remote<Server>>
  worker: Worker
  id: string = "WebWorker"

  constructor(worker: Worker) {
    this.handlers = {};
    this.proxy_lookups = new Map();
    this.worker = worker;
    const ServerClass = wrap<Server>(worker);
    this.ServerPromise = new ServerClass();
  }
  connect() {}
  on(signal: string, handler: EventCallback) {
    const proxy_handler = proxy(handler);
    this.proxy_lookups.set(handler, proxy_handler);
    return this.ServerPromise.then((server) => {
      return server.addHandler(signal, proxy(handler));
    });
  }
  off(signal: string, handler: EventCallback) {
    const proxy_handler = this.proxy_lookups.get(handler);
    if (proxy_handler !== undefined) {
      console.log("removing handler: ", handler);
      return this.ServerPromise.then((server) => {
        return server.removeHandler(signal, proxy_handler);
      })
    }
  }
  async mountLocal() {
    const dirHandle = await window.showDirectoryPicker();
    console.log({dirHandle});
    
    const server = await this.ServerPromise;
    await server.mount(dirHandle);
  }
  async asyncEmit(signal: string, ...args: any[]) {
    const server = await this.ServerPromise;
    console.log({server, signal, args});
    let last_arg = args.pop();
    if (last_arg instanceof Function) {
      last_arg = proxy(last_arg);
    }
    if (last_arg !== undefined) {
      args.push(last_arg);
    }
    return await server.onAsyncEmit(signal, ...args);
  }

}