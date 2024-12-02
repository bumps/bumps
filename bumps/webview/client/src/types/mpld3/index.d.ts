declare module "mpld3" {
  interface mpld3 {
    draw_figure(
      div_id: string,
      data: { width: number; height: number },
      process: boolean | Function,
      clearElem: boolean
    ): void;
  }
  export function draw_figure(
    div_id: string,
    data: { width: number; height: number },
    process: boolean | Function,
    clearElem: boolean
  ): void;
}

// type mpld3Func = (...args: any) => any;
