<script setup lang="ts">
import { ref } from "vue";
import TagFilter from "./ParameterTagFilter.vue";
import type { AsyncSocket } from "../asyncSocket.ts";
import { setupDrawLoop } from "../setupDrawLoop";

// const title = "Parameters";
// const active_parameter = ref("");

const props = defineProps<{
  socket: AsyncSocket;
}>();

setupDrawLoop("updated_parameters", props.socket, fetch_and_draw);

type parameter_info = {
  name: string;
  id: string;
  tags: string[];
  fittable: boolean;
  fixed: boolean;
  paths: string[];
  link: string;
  writable: boolean;
  value_str: string;
  min_str: string;
  max_str: string;
  active?: boolean;
};

// const parameters = ref<parameter_info[]>([]);
const parameters_local = ref<parameter_info[]>([]);
const tag_filter = ref<typeof TagFilter>();

async function fetch_and_draw() {
  const payload = (await props.socket.asyncEmit("get_parameters", false)) as parameter_info[];
  const pl = parameters_local.value;
  payload.forEach((p, i) => {
    if (!pl[i]?.active) {
      pl.splice(i, 1, p);
    }
  });
  pl.splice(payload.length);
}

async function editItem(event: FocusEvent, item_name: "min" | "max" | "value", index: number) {
  const new_value = (event.target as HTMLElement).innerText;
  if (validate_numeric(new_value)) {
    props.socket.asyncEmit("set_parameter", parameters_local.value[index].id, item_name, new_value);
  }
}

function validate_numeric(value: string, allow_inf: boolean = false) {
  if (allow_inf && (value === "inf" || value === "-inf")) {
    return true;
  }
  return !Number.isNaN(Number(value));
}

// async function onInactive(param) {
//   param.active = false;
//   fetch_and_draw();
// }

async function setFittable(event: MouseEvent, index: number) {
  console.debug(event, event.target, index, parameters_local.value[index].fixed);
  const parameter = parameters_local.value[index];
  props.socket.asyncEmit("set_parameter", parameter.id, "fixed", !parameter.fixed);
}
</script>

<template>
  <TagFilter ref="tag_filter" :parameters="parameters_local"></TagFilter>
  <table class="table">
    <thead class="border-bottom py-1 sticky-top text-white bg-secondary">
      <tr>
        <th scope="col">Fit?</th>
        <th scope="col">Name</th>
        <th scope="col">Value</th>
        <th scope="col">Paths</th>
      </tr>
    </thead>
    <tbody>
      <tr v-for="{ parameter: param, index } in tag_filter?.filtered_parameters" :key="param.id" class="py-1">
        <td>
          <label class="visually-hidden" for="`param-checkbox-${index}`">Fit?</label>
          <input
            v-if="param.fittable"
            id="`param-checkbox-${index}`"
            class="form-check-input"
            type="checkbox"
            :checked="!param.fixed"
            @click.prevent="setFittable($event, index)"
          />
        </td>
        <td>
          {{ param.name }}
          <div v-if="tag_filter?.show_tags">
            <span
              v-for="tag in param.tags"
              :key="`tag-${tag}`"
              class="badge rounded-pill me-1"
              :style="{ color: 'white', 'background-color': tag_filter.tag_colors[tag] }"
            >
              {{ tag }}
            </span>
          </div>
        </td>
        <td
          :contenteditable="param.writable"
          spellcheck="false"
          @blur="(e) => editItem(e, 'value', index)"
          @keydown.enter="(e) => (e.target as HTMLElement).blur()"
        >
          {{ param.value_str }}
        </td>
        <td>
          <p v-for="path in param.paths" :key="path" class="my-0">{{ path }}</p>
        </td>
      </tr>
    </tbody>
  </table>
</template>

<style scoped>
table {
  white-space: nowrap;
}
</style>
