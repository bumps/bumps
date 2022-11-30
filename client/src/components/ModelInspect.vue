<script setup lang="ts">
import { ref, onMounted, watch, onUpdated, computed, shallowRef, nextTick } from 'vue';
import JsonViewer from 'vue-json-viewer';
import { getDiff } from 'json-difference';
import type { Socket } from 'socket.io-client';

// from https://github.com/microsoft/TypeScript/issues/1897#issuecomment-1228063688
type json =
  | string
  | number
  | boolean
  | null
  | json[]
  | { [key: string]: json };

const title = "Model";
const active_parameter = ref("");

const props = defineProps<{
  socket: Socket,
  visible: Boolean
}>();

const modelJson = ref<json>();

props.socket.on('update_parameters', fetch_and_draw);

function fetch_and_draw() {
  if (props.visible) {
    props.socket.emit('get_model', (payload: json) => {
      const old_model: json = modelJson.value as Record<string, any>;
      const new_model: json = payload as Record<string, any>;
      const diff = getDiff(old_model, new_model);
      for (let [path, oldval, newval] of diff.edited) {
        const { target, parent, key } = resolve_diffpath(old_model, path);
        // trigger reactive update;
        if (typeof (key) === 'number' && Array.isArray(parent)) {
          parent.splice(key, 1, newval);
        }
        else {
          parent[key] = newval;
        }
      }
    });
  }
}

watch(() => props.visible, (value) => {
  if (value) {
    props.socket.emit('get_model', (payload: json) => {
      modelJson.value = payload;
    });
  }
});

// function* traverse(o: object, path: string[]=[]) {
//     for (var i of Object.keys(o)) {
//         const itemPath = path.concat(i);
//         yield [i,o[i],itemPath,o];
//         if (o[i] !== null && typeof(o[i])=="object") {
//             //going one step down in the object tree!!
//             yield* traverse(o[i],itemPath);
//         }
//     }
// }

// function resolve(o: object, path: string[] = []) {
//   let target = o;
//   for (let pfrag of path) {
//     target = target[pfrag];
//   }
//   return target;
// }

const ARRAY_PATH_RE = /^([0-9]+)\[\]$/;
function resolve_diffpath(o: object, diffpath: string) {
  const pathitems = diffpath.split('/');
  let target = o;
  let parent: object = o;
  let key: number | string = "";
  for (let pathitem of pathitems) {
    const array_path = pathitem.match(ARRAY_PATH_RE);
    parent = target;
    key = (array_path) ? Number(array_path[1]) : pathitem;
    target = target[key];
  }
  return { target, parent, key };
}

</script>

<template>
  <JsonViewer :value="modelJson" />
</template>