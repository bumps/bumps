<script setup lang="ts">
import { format, formatDistance, formatRelative, subDays } from 'date-fns'
import { ref, onMounted, watch, onUpdated, computed, shallowRef } from 'vue';
import { Modal } from 'bootstrap/dist/js/bootstrap.esm.js';
import type { AsyncSocket } from '../asyncSocket.ts';

const props = defineProps<{
  socket: AsyncSocket,
  title: string,
  show_files?: boolean,
  show_name_input?: boolean,
  require_name?: boolean,
  name_input_label?: string,
  chosenfile_in?: string,
  search_patterns?: string[], // comma-delimited glob patterns
  callback: (pathlist: string[], filename: string) => void,
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
const pathlist = shallowRef<string[]>(["/"]);
const subdirlist = shallowRef<FileInfo[]>([])
const filelist = shallowRef<FileInfo[]>([])
const filtered_filelist = shallowRef<FileInfo[]>([]);
const chosenFile = ref("");
const sortby = ref<sortOrder>("name");
const reversed = ref(false);
const step = ref(1);
const active_search_pattern = ref<string | null>(null);
const active_search_regexp = ref<RegExp | null>(null);

let modal: Modal;
onMounted(() => {
  modal = new Modal(dialog.value, { backdrop: 'static', keyboard: false });
});

function close() {
  modal?.hide();
  isOpen.value = false;
}

async function open() {
  await props.socket.asyncEmit('get_current_pathlist', (new_pathlist) => {
    setPath(new_pathlist);
  })
  chosenFile.value = props.chosenfile_in ?? "";
  if (props.search_patterns && props.search_patterns.length > 0) {
    active_search_pattern.value = props.search_patterns[0];
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

async function setPath(new_pathlist?: string[]) {
  pathlist.value = new_pathlist ?? [];
  await props.socket.asyncEmit("get_dirlisting", pathlist.value, ({ files, subfolders }: {files: FileInfo[], subfolders: FileInfo[]}) => {
    subdirlist.value = subfolders.sort(FileInfoSorter);
    filelist.value = files;
    filtered_filelist.value = files.filter(FileInfoSearch).sort(FileInfoSorter);
  })
}

const PREFIXES = ["", "k", "M", "G", "T", "P"];
const SCALES = PREFIXES.map((_, i) => Math.pow(1024, i));
const LOG_1024 = Math.log(1024);
function formatSize(bytes) {
  const scale = (bytes > 0) ? Math.min(5, Math.floor(Math.log(bytes) / LOG_1024)) : 0;
  const prefix = PREFIXES[scale];
  const reduced = bytes / SCALES[scale];
  const extra_digits = (scale == 0) ? 0 : 1;
  return `${reduced.toFixed(extra_digits)} ${prefix}B`;
}

async function chooseFile() {
  await props.callback(pathlist.value, chosenFile.value);
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
            <h5 class="modal-title" id="fileBrowserLabel">{{title}}</h5>
            <button type="button" class="btn-close" @click="close" aria-label="Close"></button>
        </div>
        <div class="p-1 border-bottom">
          <div class="container py-1 px-3">
            <nav aria-label="breadcrumb">
              <ol class="breadcrumb mb-0">
                <li v-for="(pathitem, index) in pathlist" :key="index" class="breadcrumb-item">
                  <a href="#" @click.prevent="setPath(pathlist.slice(0, index+1))">{{pathitem}}</a>
                </li>
              </ol>
            </nav>
          </div>
          <div v-if="show_name_input === true" class="container">
            <div class="row align-items-center mb-1">
              <div class="col-auto">
                <label for="userfilename" class="col-form-label">{{name_input_label}}:</label>
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
          <div class="container" v-if="show_files">
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
          <label v-if="search_patterns && search_patterns.length > 0">Search: 
            <select v-model="active_search_pattern">
              <option v-for="search_pattern in search_patterns" :key="search_pattern">
                {{ search_pattern }}
              </option>
              <option :value="null">All files</option>
            </select>
          </label>
          <button type="button" class="btn btn-secondary" @click="close">Cancel</button>
          <button type="button" class="btn btn-primary" :class="{disabled: require_name && chosenFile == ''}"
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