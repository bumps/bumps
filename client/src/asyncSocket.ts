import { Socket } from 'socket.io-client';

declare module 'socket.io-client' {
  class Socket {
    public asyncEmit(ev: string, ...args: any[]): Promise<any>;
  }
}

Socket.prototype.asyncEmit = async function asyncEmit(ev: string, ...args: any[]) {
  return new Promise((resolve, reject) => {
    this.emit(ev, ...args, resolve);
  })
};