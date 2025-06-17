import { Socket } from "socket.io-client";
import { decode } from "@msgpack/msgpack";

declare module "socket.io-client" {
  //
  interface Socket {
    asyncEmit(ev: string, ...args: any[]): Promise<any>;
    // public emit(event: string, ...args: any[]): void;
  }
}

Socket.prototype.asyncEmit = async function asyncEmit(ev: string, ...args: any[]) {
  const callback = args[args.length - 1] instanceof Function ? args.pop() : null;
  return new Promise((resolve) => {
    // get result of plain emit:
    this.emit(ev, ...args, resolve);
  }).then(async (result) => {
    // if the result is a buffer, decode it
    if (result instanceof ArrayBuffer) {
      result = decode(new Uint8Array(result));
    } else if (result instanceof Uint8Array) {
      result = decode(result);
    }
    // execute (possibly async) callback and then return promise of result
    if (callback !== null) {
      await callback(result);
    }
    return result;
  });
};

export class AsyncSocket extends Socket {}
