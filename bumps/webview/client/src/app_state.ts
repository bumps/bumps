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
export const active_fit = ref<{ fitter_id?: string, options?: {}, num_steps?: number }>({});
export const fit_progress = ref<{ chisq?: string, step?: number, value?: number }>({});
export const fitter_settings = shallowRef<{ [fit_name: string]: FitSetting }>({});
export const selected_fitter = ref<string>("amoeba");
export const notifications = ref<{ title: string, content: string, id: string, spinner: boolean }[]>([]);
export const menu_items = ref<{disabled?: Ref<boolean> | boolean, text: string, action: Function, help?: string}[]>([]);
export const autosave_history = ref(false);
export const autosave_history_length = ref(10);
export const session_output_file = shallowRef<{ filename: string, pathlist: string[] }>();
export const autosave_session = ref(false);
export const autosave_session_interval = ref(300);

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