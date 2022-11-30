<script setup lang="ts">
import { ref, onMounted, watch, onUpdated, computed, shallowRef, nextTick } from 'vue';
import type { Socket } from 'socket.io-client';

const title = "Summary";
const active_parameter = ref("");

const props = defineProps<{
  socket: Socket,
  visible: Boolean
}>();

type parameter_info = {
  name: string,
  id: string,
  value01: number,
  value_str: string,
  min_str: string,
  max_str: string,
  active?: boolean,
}

const parameters = ref<parameter_info[]>([]);
const parameters_local01 = ref<number[]>([]);
const parameters_localstr = ref<string[]>([]);

onMounted(() => {
  props.socket.on('update_parameters', () => {
    fetch_and_draw();
  });
});

function fetch_and_draw() {
  if (!props.visible) {
    // do nothing if not visible
    return
  }
  props.socket.emit('get_parameters', true, (payload: parameter_info[]) => {
    // console.log(payload);
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
  });
}

async function onMove(param_index) {
  // props.socket.volatile.emit('set_parameter01', parameters.value[param_index].name, parameters_local01.value[param_index]);
  props.socket.emit('set_parameter', parameters.value[param_index].id, "value01", parameters_local01.value[param_index]);
}

async function editItem(ev, item_name: "min" | "max" | "value", index: number) {
  const new_value = ev.target.innerText;
  if (validate_numeric(new_value)) {
    props.socket.emit('set_parameter', parameters.value[index].id, item_name, new_value);
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
  props.socket.emit('set_parameter', parameters.value[index].id, "value01", parameters_local01.value[index]);
}

async function onInactive(param) {
  param.active = false;
  fetch_and_draw();
}

watch(() => props.visible, (value) => {
  fetch_and_draw();
});

</script>
        
<template>
  <table class="table">
    <thead class="border-bottom py-1">
      <tr>
        <th scope="col">Fit Parameter</th>
        <th scope="col"></th>
        <th scope="col">Value</th>
        <th scope="col">Min</th>
        <th scope="col">Max</th>
      </tr>
    </thead>
    <tbody>
      <tr class="py-1" v-for="(param, index) in parameters" :key="param.id">
        <td>{{ param.name }}</td>
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