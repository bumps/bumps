import { createApp } from "vue";
import "bootstrap/dist/css/bootstrap.min.css";
import "./style.css";
import { io } from "socket.io-client";
import App from "./App.vue";
import plugins from "./global/plugins";
import { panels } from "./panels";

const urlParams = new URLSearchParams(window.location.search);
const singlePanel = urlParams.get("single_panel");
const sio_base_path = urlParams.get("base_path") ?? window.location.pathname;
const sio_server = urlParams.get("server") ?? "";
const name = "Bumps";

const socket = io(sio_server, {
  path: `${sio_base_path}socket.io`,
});

const app = createApp(App, { panels, socket, singlePanel, name });

/** register plugins/middleware */
for (const [plugin, options] of plugins) app.use(plugin, options);

app.mount("#app");
