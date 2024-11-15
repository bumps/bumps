import { ref, shallowRef } from 'vue';
import { v4 as uuidv4 } from 'uuid';
import type { Ref } from 'vue';
import type { AsyncSocket } from './asyncSocket.ts';

interface ModalDialog {
  open: (...args: unknown[]) => void,
  close: () => void,
}

export class FileBrowserSettings {
  chosenfile_in?: string
  pathlist_in?: string[]
  title: string
  show_name_input: boolean
  require_name: boolean
  name_input_label?: string
  show_files: boolean
  search_patterns: string[]
  callback: (pathlist: string[], filename: string) => Promise<void>
};

export type FitSetting = { name: string, settings: object };

export const socket = ref<AsyncSocket>();
export const connected = ref(false);
export const fitOptions = ref<ModalDialog>();
export const fileBrowser = ref<ModalDialog>();
export const model_file = shallowRef<{ filename: string, pathlist: string[] }>();
export const model_loaded = ref<string>();
export const active_layout = ref("left-right");
export const active_panel = ref([0, 1]);
export const active_fit = ref<{ fitter_id?: string, options?: {}, num_steps?: number, chisq?: string, step?: number, value?: number }>({});
export const fitter_settings = shallowRef<{ [fit_name: string]: FitSetting }>({});
export const selected_fitter = ref<string>("amoeba");
export const notifications = ref<{ title: string, content: string, id: string, spinner: boolean }[]>([]);
export const menu_items = ref<{disabled?: Ref<boolean> | boolean, text: string, action: Function, help?: string}[]>([]);
export const autosave_history = ref(false);
export const autosave_history_length = ref(10);
export const session_output_file = shallowRef<{ filename: string, pathlist: string[] }>();
export const autosave_session = ref(false);
export const autosave_session_interval = ref(300);


class SharedState {
  public updated_convergence: Ref<undefined | string> = ref(undefined);
  public updated_uncertainty: Ref<undefined | string> = ref(undefined);
  public updated_parameters: Ref<undefined | string> = ref(undefined);
  public updated_model: Ref<undefined | string> = ref(undefined);
  public updated_history: Ref<undefined | string> = ref(undefined);
  public selected_fitter: Ref<undefined | string> = ref(undefined);
  public fitter_settings: Ref<undefined | {[fit_name: string]: FitSetting}> = ref(undefined);
  public active_fit: Ref<undefined | { fitter_id?: string, options?: {}, num_steps?: number, chisq?: string, step?: number, value?: number }> = ref(undefined);
  public model_file: Ref<undefined | { filename: string, pathlist: string[] }> = ref(undefined);
  public model_loaded: Ref<undefined | string> = ref(undefined);
  public session_output_file: Ref<undefined | { filename: string, pathlist: string[] }> = ref(undefined);
  public autosave_session: Ref<boolean> = ref(false);
  public autosave_session_interval: Ref<number> = ref(300);
  public autosave_history: Ref<boolean> = ref(true);
  public autosave_history_length: Ref<number> = ref(10);
  public uncertainty_available: Ref<undefined | object> = ref(undefined);
  public custom_plots_available: Ref<undefined | object> = ref(undefined);
}

export class AutoupdateState {
  value: SharedState;
  constructor() {
    this.value = new SharedState();
    // this.value.active_fit.value = undefined;
    // this.value.autosave_history.value = true;
    // this.value.autosave_history_length.value = 10;
    // this.value.autosave_session.value = false;
    // this.value.autosave_session_interval.value = 300;
    // this.value.custom_plots_available.value = undefined;
    // this.value.fitter_settings.value = undefined;
    // this.value.model_file.value = undefined;
    // this.value.model_loaded.value = undefined;
    // this.value.selected_fitter.value = undefined;
    // this.value.session_output_file.value = undefined;
    // this.value.uncertainty_available.value = undefined;
    // this.value.updated_convergence.value = undefined;
    // this.value.updated_history.value = undefined;
    // this.value.updated_model.value = undefined;
    // this.value.updated_parameters.value = undefined;
    // this.value.updated_uncertainty.value = undefined;
  }

  async init(socket: AsyncSocket) {
    for (const key in this.value) {
      const initial_value = await socket.asyncEmit(`get_shared_setting`, key);
      this.value[key].value = initial_value;
    }

    for (const key in this.value) {
      socket.on(key, (value) => {
        console.log(`Received update for ${key}: ${JSON.stringify(value, null, 2)}`);
        this.value[key].value = value;
        console.log(`Updated ${key}`, this.value[key].value);
      });
    }
  }
}

export const shared_state = new AutoupdateState();

export function cancelNotification(id: string) {
  const index = notifications.value.findIndex(({id: item_id}) => (item_id === id));
  if (index > -1) {
    notifications.value.splice(index, 1);
  }
}

export function addNotification({ title, content, timeout, id }: {title: string, content: string, timeout?: number, id?: string}) {
  const has_timeout = (timeout !== undefined);
  id = id || uuidv4();
  notifications.value.push({title, content, id, spinner: !has_timeout});
  if (has_timeout) {
    setTimeout(cancelNotification, timeout, id);
  }
  return id;
}