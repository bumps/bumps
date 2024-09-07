<script setup lang="ts">
import { format, formatDistance, formatRelative, subDays } from 'date-fns'
import { ref, onMounted, watch, onUpdated, computed, shallowRef } from 'vue';
import { Modal } from 'bootstrap/dist/js/bootstrap.esm.js';
import { addNotification } from '../app_state';
import type { AsyncSocket } from '../asyncSocket.ts';
import type { FileBrowserSettings } from '../app_state';

const props = defineProps<{
  socket: AsyncSocket
}>();

const emit = defineEmits<{
  (e: 'selected', pathlist: string[], filename: string): void
}>();

interface FileInfo  {
  name: string,
  size: number,
  modified: number
}

type sortOrder = "name" | "size" | "modified";

const UP_ARROW = "▲";
const DOWN_ARROW = "▼";

const dialog = ref<HTMLDivElement>();
const isOpen = ref(false);
const pathlist = shallowRef<string[]>([]);
const subdirlist = shallowRef<FileInfo[]>([]);
const filelist = shallowRef<FileInfo[]>([]);
const drives = ref<string[]>([]);
const filtered_filelist = shallowRef<FileInfo[]>([]);
const chosenFile = ref("");
const sortby = ref<sortOrder>("name");
const reversed = ref(false);
const step = ref(1);
const active_search_pattern = ref<string | null>(null);
const active_search_regexp = ref<RegExp | null>(null);
const settings = ref<FileBrowserSettings>();

let modal: Modal;
onMounted(() => {
  modal = new Modal(dialog.value as HTMLElement, { backdrop: 'static', keyboard: false });
});

function close() {
  modal?.hide();
  isOpen.value = false;
}

async function open(settings_in: FileBrowserSettings) {
  settings.value = settings_in;
  chosenFile.value = settings_in.chosenfile_in ?? "";
  if (settings_in.pathlist_in) {
    pathlist.value = settings_in.pathlist_in;
  }
  await setPath(pathlist.value ?? []);
  if (settings_in.search_patterns && settings_in.search_patterns.length > 0) {
    // this triggers doSorting and filtering...
    active_search_pattern.value = settings_in.search_patterns[0];
  }
  modal?.show();
  isOpen.value = true;
}

function FileInfoSorter(a: FileInfo, b: FileInfo) {
  if (a[sortby.value] > b[sortby.value]) {
    return step.value;
  }
  else if (a[sortby.value] < b[sortby.value]) {
    return -step.value;
  }
  else {
    return 0
  }
}

function FileInfoSearch(f: FileInfo) {
  if (active_search_regexp.value === null) {
    return true;
  }
  else {
    return active_search_regexp.value.test(f.name);
  }
}

async function subdirClick(subdir: string) {
  pathlist.value.push(subdir);
  await setPath(pathlist.value);
}

interface DirListing {
  drives: string[],
  pathlist: string[],
  files: FileInfo[],
  subfolders: FileInfo[]
}

interface DirListingError {
  error: string
}

props.socket.on("base_path", (pathlist_in: string[]) => {
  pathlist.value = pathlist_in;
})

async function setPath(new_pathlist?: string[]) {
  let result = await props.socket.asyncEmit("get_dirlisting", new_pathlist) as DirListing | DirListingError;
  if ('error' in result) {
    addNotification({ title: result.error, content: "reverting to base path", timeout: 3000 });
    result = await props.socket.asyncEmit("get_dirlisting", null) as DirListing;
  }
  const { drives: drives_in, pathlist: abs_pathlist, files, subfolders } = result;
  subdirlist.value = subfolders.sort(FileInfoSorter);
  filelist.value = files;
  filtered_filelist.value = files.filter(FileInfoSearch).sort(FileInfoSorter);
  // server include absolute pathlist in response...
  pathlist.value = abs_pathlist;
  drives.value = drives_in;
}

const PREFIXES = ["", "k", "M", "G", "T", "P"];
const SCALES = PREFIXES.map((_, i) => Math.pow(1024, i));
const LOG_1024 = Math.log(1024);
function formatSize(bytes: number) {
  const scale = (bytes > 0) ? Math.min(5, Math.floor(Math.log(bytes) / LOG_1024)) : 0;
  const prefix = PREFIXES[scale];
  const reduced = bytes / SCALES[scale];
  const extra_digits = (scale == 0) ? 0 : 1;
  return `${reduced.toFixed(extra_digits)} ${prefix}B`;
}

async function chooseFile() {
  await props.socket.asyncEmit("set_base_path", pathlist.value);
  await settings.value?.callback(pathlist.value, chosenFile.value);
  close();
}

function doSorting() {
  subdirlist.value.sort(FileInfoSorter);
  filtered_filelist.value = filelist.value.filter(FileInfoSearch).sort(FileInfoSorter);
}

function toggleSorting(column: sortOrder) {
  if (sortby.value === column) {
    // toggling direction of already-selected column
    step.value = -step.value;
  }
  else {
    sortby.value = column;
    step.value = 1;
  }
  doSorting();
}

function calculateIcon(column: sortOrder) {
  if (sortby.value === column) {
    return (step.value > 0) ? UP_ARROW : DOWN_ARROW;
  }
  else {
    return "";
  }
}

const SUFFIX_SEARCH = /^(.*)(\.[^\.]+)$/
function selectNotSuffix(ev: Event) {
  const target = ev.target as HTMLInputElement;
  const value = target.value;
  const matches = value.match(SUFFIX_SEARCH);
  if (matches !== null) {
    const selection_length = matches[1].length;
    target.setSelectionRange(0, selection_length);
  }
}

watch(active_search_pattern, async (newval, oldval) => {
  if (newval === null) {
    active_search_regexp.value = null;
  }
  else {
    const subexps = newval?.split(',') ?? [];
    const re = subexps.map((s) => `(${s.trim().replace(/\./g, '\.')})`);
    active_search_regexp.value = new RegExp(`(${re.join('|')})$`);
  }
  doSorting();
})

defineExpose({
  close,
  open
})
</script>

<template>
  <div ref="dialog" class="modal fade" id="fileBrowserModal" tabindex="-1" aria-labelledby="fileBrowserLabel"
    :aria-hidden="isOpen">
    <div class="modal-dialog modal-lg modal-dialog-scrollable">
      <div class="modal-content">
        <div class="modal-header">
            <h5 class="modal-title" id="fileBrowserLabel">{{settings?.title}}</h5>
            <button type="button" class="btn-close" @click="close" aria-label="Close"></button>
        </div>
        <div class="p-1 border-bottom">
          <div class="container py-1 px-3" v-if="drives.length > 1">
            <nav aria-label="breadcrumb">
              <ol class="breadcrumb mb-0">
                <li v-for="drive in drives" :key="drive" class="breadcrumb-item">
                  <a href="#" @click.prevent="setPath([drive])">{{drive}}</a>
                </li>
              </ol>
            </nav>
          </div>
          <div class="container py-1 px-3">
            <nav aria-label="breadcrumb">
              <ol class="breadcrumb mb-0">
                <li v-for="(pathitem, index) in pathlist" :key="index" class="breadcrumb-item">
                  <a href="#" @click.prevent="setPath(pathlist.slice(0, index+1))">{{pathitem}}</a>
                </li>
              </ol>
            </nav>
          </div>
          <div v-if="settings?.show_name_input === true" class="container">
            <div class="row align-items-center mb-1">
              <div class="col-auto">
                <label for="userfilename" class="col-form-label">{{settings?.name_input_label}}:</label>
              </div>
              <div class="col">
                <input 
                  type="text"
                  id="userfilename"
                  class="form-control"
                  v-model="chosenFile"
                  @focus="selectNotSuffix"
                  @keyup.enter="chooseFile"
                  >
              </div>
            </div>
          </div>
        </div>
        <div class="modal-body">
          <div class="container border-bottom">
            <h5>Subdirectories:</h5>
            <div class="row row-cols-3">
              <div class="col overflow-hidden" v-for="subdir in subdirlist" :key="subdir.name">
                <a href="#" @click.prevent="subdirClick(subdir.name)" :title="subdir.name">{{subdir.name}}</a>
              </div>
            </div>
          </div>
          <div>show files: {{ settings?.show_files }}</div>
          <div class="container" v-if="settings?.show_files">
            <h5>Files:</h5>
            <table class="table table-sm">
              <thead>
                <tr class="sticky-top text-body bg-white">
                  <th scope="col" @click="toggleSorting('name')">Name{{ calculateIcon('name') }}</th>
                  <th scope="col" @click="toggleSorting('size')">Size{{ calculateIcon('size') }}</th>
                  <th scope="col" @click="toggleSorting('modified')">Modified{{ calculateIcon('modified') }}</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="fileinfo in filtered_filelist" 
                  :key="fileinfo.name"
                  @click="chosenFile=fileinfo.name"
                  :title="fileinfo.name"
                  @dblclick="chosenFile=fileinfo.name;chooseFile()"
                  :class="{'table-warning': fileinfo.name == chosenFile}"
                  > 
                  <td>{{ fileinfo.name }}</td>
                  <td>{{ formatSize(fileinfo.size) }}</td>
                  <td>{{ formatRelative(new Date(fileinfo.modified * 1000), new Date()) }}</td>
                </tr>
              </tbody>
            </table>
          </div>  
        </div>
        <div class="modal-footer">
          <label v-if="settings?.search_patterns && settings?.search_patterns.length > 0">Search: 
            <select v-model="active_search_pattern">
              <option v-for="search_pattern in settings?.search_patterns ?? []" :key="search_pattern">
                {{ search_pattern }}
              </option>
              <option :value="null">All files</option>
            </select>
          </label>
          <button type="button" class="btn btn-secondary" @click="close">Cancel</button>
          <button type="button" class="btn btn-primary" :class="{disabled: settings?.require_name && chosenFile == ''}"
            @click="chooseFile">OK</button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.active {
  background-color: light
}

table tbody tr {
  cursor: pointer;
}
</style>