import { ref, shallowRef } from 'vue';
import type { Ref } from 'vue';
import type { AsyncSocket } from './asyncSocket';
import { v4 as uuidv4 } from 'uuid';

interface ModalDialog {
  open: () => void,
  close: () => void,
}
export type FitSetting = { name: string, settings: object };

export const socket = ref<AsyncSocket>();
export const connected = ref(false);
export const fitOptions = ref<ModalDialog>();
export const fileBrowser = ref<ModalDialog>();
export const fileBrowserSettings = ref({
  chosenfile_in: "",
  title: "",
  show_name_input: false,
  require_name: false,
  name_input_label: "",
  show_files: true,
  search_patterns: [""],
  callback: (pathlist: string[], filename: string) => { },
});
export const model_loaded = shallowRef<{ pathlist: string[], filename: string }>();
export const active_layout = ref("left-right");
export const active_panel = ref([0, 1]);
export const fit_active = ref<{ fitter_id?: string, options?: {}, num_steps?: number }>({});
export const fit_progress = ref<{ chisq?: string, step?: number, value?: number }>({});
export const fitter_settings = shallowRef<{ [fit_name: string]: FitSetting }>({});
export const fitter_active = ref<string>("amoeba");
export const notifications = ref<{ title: string, content: string, id: string, spinner: boolean }[]>([]);
export const menu_items = ref<{disabled?: Ref<boolean>, text: string, action: Function, help?: string}[]>([]);

export function add_notification({title, content, id, timeout}: {title: string, content: string, id?: string, timeout?: number}) {
  if (id === undefined) {
    id = uuidv4();
  }
  const has_timeout = (timeout !== undefined);
  notifications.value.push({title, content, id, spinner: !has_timeout});
  if (has_timeout) {
    setTimeout(cancelNotification, timeout, id);
  }
}

export function cancelNotification(id: string) {
  const index = notifications.value.findIndex(({id: item_id}) => (item_id === id));
  if (index > -1) {
    notifications.value.splice(index, 1);
  }
}