declare module "mpld3" {
  interface mpld3 {
    draw_figure(): void;
  }

  export function draw_figure(
    div_id: string,
    data: { width: number; height: number },
    process?: false | ((fig: any, element: HTMLElement) => void),
    clearElem?: boolean
  ): void;
}
