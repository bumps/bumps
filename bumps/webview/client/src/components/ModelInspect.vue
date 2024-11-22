<script setup lang="ts">
import { ref, onMounted } from 'vue';
import { setupDrawLoop } from '../setupDrawLoop';
import VueJsonPretty from 'vue-json-pretty';
import { getDiff } from 'json-difference';
import type { AsyncSocket } from '../asyncSocket.ts';
import 'vue-json-pretty/lib/styles.css';


// from https://github.com/microsoft/TypeScript/issues/1897#issuecomment-1228063688
type json =
  | string
  | number
  | boolean
  | null
  | json[]
  | { [key: string]: json };

const title = "Model";
// @ts-ignore: intentionally infinite type recursion
const modelJson = ref<json>({});

const props = defineProps<{
  socket: AsyncSocket,
}>();

setupDrawLoop('updated_parameters', props.socket, fetch_and_draw);

props.socket.on('model_loaded', () => { fetch_and_draw('', true) });

async function fetch_and_draw(timestamp: string = '', reset: boolean = false) {
  const payload = await props.socket.asyncEmit('get_model') as ArrayBuffer;
  const json_bytes = new Uint8Array(payload);
  const json_value: json = JSON.parse(decoder.decode(json_bytes));
  if (reset) {
    modelJson.value = json_value;
  }
  else {
    // do update of existing model:
    const old_model: json = modelJson.value as Record<string, any>;
    const new_model: json = json_value as Record<string, any>;
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
  }
}

const decoder = new TextDecoder('utf-8');

onMounted(() => {
  fetch_and_draw('', true);
})

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
  <vue-json-pretty :data="modelJson" :deep="1" :showLength="true" />
</template>