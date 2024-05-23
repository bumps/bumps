import { ref, shallowRef } from 'vue';
import type { Ref } from 'vue';
import type { AsyncSocket } from './asyncSocket.ts';

interface ModalDialog {
  open: () => void,
  close: () => void,
}

interface FileBrowserSettings {
  chosenfile_in: string,
  pathlist_in: string[],
  title: string,
  show_name_input: boolean,
  require_name: boolean,
  name_input_label: string,
  show_files: boolean,
  search_patterns: string[],
  callback: (pathlist: string[], filename: string) => void,
};

export type FitSetting = { name: string, settings: object };

export const socket = ref<AsyncSocket>();
export const connected = ref(false);
export const fitOptions = ref<ModalDialog>();
export const fileBrowser = ref<ModalDialog>();
export const fileBrowserSettings = ref<FileBrowserSettings>({
  chosenfile_in: "",
  pathlist_in: [],
  title: "",
  show_name_input: false,
  require_name: false,
  name_input_label: "",
  show_files: true,
  search_patterns: [""],
  callback: (pathlist: string[], filename: string) => { },
});
export const model_filename = ref<string>();
export const model_pathlist = ref<string[]>([]);
export const model_loaded = ref<string>();
export const active_layout = ref("left-right");
export const active_panel = ref([0, 1]);
export const active_fit = ref<{ fitter_id?: string, options?: {}, num_steps?: number }>({});
export const fit_progress = ref<{ chisq?: string, step?: number, value?: number }>({});
export const fitter_settings = shallowRef<{ [fit_name: string]: FitSetting }>({});
export const selected_fitter = ref<string>("amoeba");
export const notifications = ref<{ title: string, content: string, id: string, spinner: boolean }[]>([]);
export const menu_items = ref<{disabled?: Ref<boolean> | boolean, text: string, action: Function, help?: string}[]>([]);