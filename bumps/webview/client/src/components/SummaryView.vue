<script setup lang="ts">
import { ref, computed } from 'vue';
import type { AsyncSocket } from '../asyncSocket.ts';
import { setupDrawLoop} from '../setupDrawLoop';
import TagFilter from './ParameterTagFilter.vue'

const title = "Summary";

const props = defineProps<{
  socket: AsyncSocket,
}>();

setupDrawLoop('updated_parameters', props.socket, fetch_and_draw, title);

type parameter_info = {
  name: string,
  id: string,
  tags: string[],
  value01: number,
  value_str: string,
  min_str: string,
  max_str: string,
  active?: boolean,
}

const parameters = ref<parameter_info[]>([]);
const parameters_local01 = ref<number[]>([]);
const parameters_localstr = ref<string[]>([]);
const show_tags = ref(false);
const tag_filter = ref<typeof TagFilter>();

async function fetch_and_draw() {
  const payload: parameter_info[] = await props.socket.asyncEmit('get_parameters', true) as parameter_info[];
  const pv = parameters.value;
  payload.forEach((p, i) => {
    parameters_localstr.value[i] = p.value_str;
    if (!(pv[i]?.active)) {
      pv.splice(i, 1, p);
      parameters_local01.value[i] = p.value01;
    }
  });
  pv.splice(payload.length);
  parameters_local01.value.splice(payload.length);
  parameters_localstr.value.splice(payload.length);
}

async function onMove(param_index) {
  // props.socket.volatile.emit('set_parameter01', parameters.value[param_index].name, parameters_local01.value[param_index]);
  props.socket.asyncEmit('set_parameter', parameters.value[param_index].id, "value01", parameters_local01.value[param_index]);
}

async function editItem(ev, item_name: "min" | "max" | "value", index: number) {
  const new_value = ev.target.innerText;
  if (validate_numeric(new_value)) {
    props.socket.asyncEmit('set_parameter', parameters.value[index].id, item_name, new_value);
  }
}

function validate_numeric(value: string, allow_inf: boolean = false) {
  if (allow_inf && (value === "inf" || value === "-inf")) {
    return true
  }
  return !Number.isNaN(Number(value))
}

async function scrollParam(ev, index) {
  const sign = Math.sign(ev.deltaY);
  parameters_local01.value[index] -= 0.01 * sign;
  props.socket.asyncEmit('set_parameter', parameters.value[index].id, "value01", parameters_local01.value[index]);
}

async function onInactive(param) {
  param.active = false;
  fetch_and_draw();
}

</script>
        
<template>
  <TagFilter ref="tag_filter" :parameters="parameters"></TagFilter>
  <table class="table table-sm">
    <thead class="border-bottom py-1 sticky-top text-white bg-secondary">
      <tr>
        <th scope="col">Fit Parameter</th>
        <th scope="col"></th>
        <th scope="col">Value</th>
        <th scope="col">Min</th>
        <th scope="col">Max</th>
      </tr>
    </thead>
    <tbody>
      <tr class="py-1" v-for="(param, index) in tag_filter?.filtered_parameters" :key="param.id">
        <td>{{ param.name }}
          <span 
            v-if="tag_filter?.show_tags"
            v-for="tag in param.tags"
            class="badge rounded-pill me-1" 
            :style="{color: 'white', 'background-color': tag_filter.tag_colors[tag]}"
            >
            {{ tag }}
          </span>
        </td>
        <td>
          <input type="range" class="form-range" min="0" max="1.0" step="0.005"
            v-model.number="parameters_local01[index]" @mousedown="param.active = true" @input="onMove(index)"
            @change="onInactive(param)" @wheel="scrollParam($event, index)" />
        </td>
        <td class="editable" contenteditable="true" spellcheck="false" @blur="editItem($event, 'value', index)"
          @keydown.enter="$event?.target?.blur()">{{ parameters_localstr[index] }}</td>
        <td class="editable" contenteditable="true" spellcheck="false" @blur="editItem($event, 'min', index)"
          @keydown.enter="$event?.target?.blur()">{{ param.min_str }}</td>
        <td class="editable" contenteditable="true" spellcheck="false" @blur="editItem($event, 'max', index)"
          @keydown.enter="$event?.target?.blur()">{{ param.max_str }}</td>
      </tr>
    </tbody>
  </table>
</template>
    
<style scoped>
svg {
  width: 100%;
  white-space: nowrap;
}

td.editable {
  min-width: 5em;
}


td > input {
  min-width: 5em;
}
</style>