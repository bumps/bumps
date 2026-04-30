/// <reference types="@types/plotly.js" />
/// <reference types="@types/uuid"/>
/// <reference types="vite-svg-loader" />
/// <reference types="./types/mpld3" />

declare module "*.vue" {
  import type { DefineComponent } from "vue";
  const component: DefineComponent<object, object, unknown>;
  export default component;
}
