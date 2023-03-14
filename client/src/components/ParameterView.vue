<script setup lang="ts">
import { ref } from 'vue';
import type { Socket } from 'socket.io-client';
import { setupDrawLoop } from '../setupDrawLoop';

const title = "Parameters";
const active_parameter = ref("");

const props = defineProps<{
  socket: Socket,
}>();

setupDrawLoop('update_parameters', props.socket, fetch_and_draw);

type parameter_info = {
  name: string,
  id: string,
  fittable: boolean,
  fixed: boolean,
  paths: string[],
  link: string,
  writable: boolean,
  value_str: string,
  min_str: string,
  max_str: string,
  active?: boolean,
}

const parameters = ref<parameter_info[]>([]);
const parameters_local = ref<parameter_info[]>([]);

async function fetch_and_draw() {
  const payload = await props.socket.asyncEmit('get_parameters', false);
  const pl = parameters_local.value;
  payload.forEach((p, i) => {
    if (!(pl[i]?.active)) {
      pl.splice(i, 1, p);
    }
  });
  pl.splice(payload.length);
}

async function editItem(ev, item_name: "min" | "max" | "value", index: number) {
  const new_value = ev.target.innerText;
  if (validate_numeric(new_value)) {
    props.socket.emit('set_parameter', parameters_local.value[index].id, item_name, new_value);
  }
}

function validate_numeric(value: string, allow_inf: boolean = false) {
  if (allow_inf && (value === "inf" || value === "-inf")) {
    return true
  }
  return !Number.isNaN(Number(value))
}

async function onInactive(param) {
  param.active = false;
  fetch_and_draw();
}

async function setFittable(ev, index) {
  console.log(ev, ev.target, index, parameters_local.value[index].fixed);
  const parameter = parameters_local.value[index];
  props.socket.emit('set_parameter', parameter.id, "fixed", !parameter.fixed);
}

</script>
        
<template>
  <table class="table">
    <thead class="border-bottom py-1">
      <tr>
        <th scope="col">Fit?</th>
        <th scope="col">Name</th>
        <th scope="col">Value</th>
        <th scope="col">Paths</th>
      </tr>
    </thead>
    <tbody>
      <tr class="py-1" v-for="(param, index) in parameters_local" :key="param.id">
        <td>
          <input class="form-check-input" v-if="param.fittable" type="checkbox" :checked="!param.fixed"
            @click.prevent="setFittable($event, index)" />
        </td>
        <td>{{ param.name }}</td>
        <td :contenteditable="param.writable" spellcheck="false" @blur="editItem($event, 'value', index)"
          @keydown.enter="$event?.target?.blur()">{{ param.value_str }}
        </td>
        <td>
          <p class="my-0" v-for="path in param.paths" :key="path">{{ path }}</p>
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