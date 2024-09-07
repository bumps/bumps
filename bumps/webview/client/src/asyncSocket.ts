import { Socket } from 'socket.io-client';

declare module 'socket.io-client' {
  class Socket {
    public asyncEmit(ev: string, ...args: any[]): Promise<any>;
    public syncFS(): Promise<void>;
  }
}

Socket.prototype.asyncEmit = async function asyncEmit(ev: string, ...args: any[]) {
  const callback = (args[args.length-1] instanceof Function) ? args.pop() : null;
  return new Promise((resolve, reject) => {
    // get result of plain emit:
    this.emit(ev, ...args, resolve);
  }).then( async (result) =>  {
    // execute (possibly async) callback and then return promise of result
    if (callback !== null) {
      await callback(result);
    }
    return result;
  })
};

Socket.prototype.syncFS = async function syncFS() {}

export class AsyncSocket extends Socket {};