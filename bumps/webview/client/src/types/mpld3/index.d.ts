declare module "mpld3" {
  interface mpld3 {
    draw_figure(): void;
  }

  export function draw_figure(
    div_id: string,
    data: { width: number; height: number },
    process: boolean | Function = false,
    clearElem: boolean = true
  ): void;
}
