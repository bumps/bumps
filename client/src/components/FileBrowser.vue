<script setup lang="ts">
import { ref, onMounted, watch, onUpdated, computed, shallowRef } from 'vue';
import { Modal } from 'bootstrap/dist/js/bootstrap.esm';
import { Socket } from 'socket.io-client';
import { emitValidationErrors } from 'mapbox-gl/dist/mapbox-gl-unminified';

const props = defineProps<{
  socket: Socket,
  title: string,
  callback: (pathlist: string[], filename: string) => void
}>();

const emit = defineEmits<{
  (e: 'selected', pathlist: string[], filename: string): void
}>();

const dialog = ref<HTMLDivElement>();
const isOpen = ref(false);
const pathlist = ref(["/"]);
// const pathlist = ref(["/", "Users", "bbm", "dev", "refl1d-modelbuilder"]);
const subdirlist = shallowRef<string[]>([])
const filelist = shallowRef<string[]>([])
const chosenFile = ref("");

props.socket.on('local_file_path', ({message: new_pathlist, timestamp}) => {
  pathlist.value = new_pathlist;
})

let modal: Modal;
onMounted(() => {
  modal = new Modal(dialog.value, { backdrop: 'static', keyboard: false });
});

function close() {
  modal?.hide();
  isOpen.value = false;
}

function open() {
  modal?.show();
  isOpen.value = true;
  setPath();
}

function subdirClick(subdir: string) {
  pathlist.value.push(subdir);
  setPath();
  props.socket.emit("get_dirlisting", pathlist.value, ({ files, subfolders }) => {

  })
}

function setPath(new_pathlist?: string[]) {
  if (new_pathlist !== undefined) {
    pathlist.value = new_pathlist;
    chosenFile.value = "";
  }
  props.socket.emit("get_dirlisting", pathlist.value, ({ files, subfolders }) => {
    subdirlist.value = subfolders.sort();
    filelist.value = files.sort();
  })
}

function chooseFile() {
  props.callback(pathlist.value, chosenFile.value);
  close();
}

defineExpose({
  close,
  open
})
</script>

<template>
  <div ref="dialog" class="modal fade" id="fileBrowserModal" tabindex="-1" aria-labelledby="fileBrowserLabel"
    :aria-hidden="isOpen">
    <div class="modal-dialog modal-lg">
      <div class="modal-content">
        <div class="modal-header">
          <h5 class="modal-title" id="fileBrowserLabel">{{title}}</h5>
          <button type="button" class="btn-close" @click="close" aria-label="Close"></button>
        </div>
        <div class="modal-body">
          <div class="container border-bottom">
            <nav aria-label="breadcrumb">
              <ol class="breadcrumb">
                <li v-for="(pathitem, index) in pathlist" :key="index" class="breadcrumb-item">
                  <a href="#" @click.prevent="setPath(pathlist.slice(0, index+1))">{{pathitem}}</a>
                </li>
              </ol>
            </nav>
          </div>
          <div class="container border-bottom">
            <h3>Subdirectories:</h3>
            <div class="row row-cols-3">
              <div class="col overflow-hidden" v-for="subdir in subdirlist" :key="subdir">
                <a href="#" @click.prevent="subdirClick(subdir)" :title="subdir">{{subdir}}</a>
              </div>
            </div>
          </div>
          <div class="container border-bottom">
            <h3>Files:</h3>
            <div class="row row-cols-3">
              <div class="btn col overflow-hidden border" :class="{'btn-warning': filename === chosenFile}"
                v-for="filename in filelist" :key="filename" @click="chosenFile = filename" :title="filename"
                @dblclick="chosenFile=filename;chooseFile()">
                {{filename}}
              </div>
            </div>
          </div>
          <div class="modal-footer">
            <button type="button" class="btn btn-secondary" @click="close">Cancel</button>
            <button type="button" class="btn btn-primary" :class="{disabled: chosenFile==''}"
              @click="chooseFile">OK</button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.active {
  background-color: light
}
</style>