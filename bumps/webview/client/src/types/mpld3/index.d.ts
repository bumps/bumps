declare module "mpld3" {
  interface mpld3 {
    drawFigure(
      div_id: string,
      data: { width: number; height: number },
      process: boolean | Function,
      clearElem: boolean
    ): void;
  }
  export function drawFigure(
    div_id: string,
    data: { width: number; height: number },
    process: boolean | Function,
    clearElem: boolean
  ): void;
}

// type mpld3Func = (...args: any) => any;
