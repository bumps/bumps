import { readonly, ref, shallowRef } from "vue";
import type { Ref, ShallowRef } from "vue";
import { v4 as uuidv4 } from "uuid";
import type { AsyncSocket } from "./asyncSocket.ts";

interface ModalDialog {
  open: (...args: unknown[]) => void;
  close: () => void;
}

type FileBrowserCallback = (pathlist: string[], filename: string) => Promise<void>;

export class FileBrowserSettings {
  chosenfile_in?: string;
  pathlist_in?: string[];
  title!: string;
  show_name_input!: boolean;
  require_name!: boolean;
  name_input_label?: string;
  show_files!: boolean;
  search_patterns!: string[];
  callback!: FileBrowserCallback;
}

export const LAYOUTS = ["left-right", "top-bottom", "full"];
export type FitSetting = { name: string; settings: object };

export const socket = ref<AsyncSocket>();
export const connecting = ref(false);
export const disconnected = ref(false);
export const fitOptions = ref<ModalDialog>();
export const fileBrowser = ref<ModalDialog>();
export const active_layout = ref<(typeof LAYOUTS)[number]>("left-right");
export const startPanel = ref([0, 1]);
export const default_fitter_settings = shallowRef<{ [fit_name: string]: FitSetting }>({});
export const default_fitter = "amoeba";
export const notifications = ref<{ title: string; content: string; id: string; spinner: boolean }[]>([]);
type FileMenuAction = (...args: any[]) => void;
export const file_menu_items = shallowRef<
  { disabled?: Ref<boolean>; text: string; action?: FileMenuAction; help?: string }[]
>([]);

interface ActiveFit {
  fitter_id: string;
  options: object;
  num_steps: number;
  chisq: string;
  step: number;
  value: number;
}

class SharedState {
  public updated_convergence: Ref<undefined | string> = ref(undefined);
  public updated_uncertainty: Ref<undefined | string> = ref(undefined);
  public updated_parameters: Ref<undefined | string> = ref(undefined);
  public updated_model: Ref<string | undefined> = ref(undefined);
  public updated_history: Ref<undefined | string> = ref(undefined);
  public selected_fitter: Ref<undefined | string> = ref(undefined);
  public fitter_settings: Ref<undefined | { [fit_name: string]: FitSetting }> = ref(undefined);
  public active_fit: Ref<ActiveFit | undefined> = ref(undefined);
  public model_file: Ref<undefined | { filename: string; pathlist: string[] }> = ref(undefined);
  public model_loaded: Ref<undefined | string> = ref(undefined);
  public session_output_file: ShallowRef<undefined | { filename: string; pathlist: string[] }> = shallowRef(undefined);
  public autosave_session: Ref<boolean> = ref(false);
  public autosave_session_interval: Ref<number> = ref(300);
  public autosave_history: Ref<boolean> = ref(true);
  public autosave_history_length: Ref<number> = ref(10);
  public uncertainty_available: Ref<undefined | { available: boolean; num_points: number }> = ref(undefined);
  public population_available: Ref<undefined | boolean> = ref(undefined);
  public custom_plots_available: Ref<undefined | { parameter_based: boolean; uncertainty_based: boolean }> =
    ref(undefined);
  public active_history: Ref<undefined | null | string> = ref(undefined);
}

export class AutoupdateState {
  shared_state: SharedState;
  constructor() {
    this.shared_state = new SharedState();
  }

  async init(socket: AsyncSocket) {
    const shared_state_keys = Object.keys(this.shared_state) as (keyof SharedState)[];
    for (const key of shared_state_keys) {
      const initial_value = await socket.asyncEmit(`get_shared_setting`, key);
      this.shared_state[key].value = initial_value;
    }

    for (const key of shared_state_keys) {
      socket.on(key, (value) => {
        console.debug(`Received update for ${key}: ${JSON.stringify(value, null, 2)}`);
        this.shared_state[key].value = value;
        console.debug(`Updated ${key}`, this.shared_state[key].value);
      });
    }
  }
}

export const autoupdate_state = new AutoupdateState();
export const shared_state = readonly(autoupdate_state.shared_state);

export function cancelNotification(id: string) {
  const index = notifications.value.findIndex(({ id: item_id }) => item_id === id);
  if (index > -1) {
    notifications.value.splice(index, 1);
  }
}

export function addNotification({
  title,
  content,
  timeout,
  id,
}: {
  title: string;
  content: string;
  timeout?: number;
  id?: string;
}) {
  const has_timeout = timeout !== undefined;
  id = id || uuidv4();
  notifications.value.push({ title, content, id, spinner: !has_timeout });
  if (has_timeout) {
    setTimeout(cancelNotification, timeout, id);
  }
  return id;
}
