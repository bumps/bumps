<script setup lang="ts">
import { ref, onMounted, computed, shallowRef } from 'vue';
import { Modal } from 'bootstrap/dist/js/bootstrap.esm.js';
import { fitter_settings, selected_fitter } from '../app_state.ts';
import type { FitSetting } from '../app_state.ts';
import type { AsyncSocket } from '../asyncSocket.ts';

const props = defineProps<{socket: AsyncSocket}>();

const dialog = ref<HTMLDivElement>();
const isOpen = ref(false);
const fitter_defaults = shallowRef<{ [fit_name: string]: FitSetting }>({});
const selected_fitter_local = ref("amoeba");

const FIT_FIELDS = {
  'starts': ['Starts', 'integer'],
  'steps': ['Steps', 'integer'],
  'samples': ['Samples', 'integer'],
  'xtol': ['x tolerance', 'float'],
  'ftol': ['f(x) tolerance', 'float'],
  'alpha': ['Convergence', 'float'],
  'stop': ['Stopping criteria', 'string'],
  'thin': ['Thinning', 'integer'],
  'burn': ['Burn-in steps', 'integer'],
  'pop': ['Population', 'float'],
  'init': ['Initializer', ["eps", "lhs", "cov", "random"]],
  'CR': ['Crossover ratio', 'float'],
  'F': ['Scale', 'float'],
  'nT': ['# Temperatures', 'integer'],
  'Tmin': ['Min temperature', 'float'],
  'Tmax': ['Max temperature', 'float'],
  'radius': ['Simplex radius', 'float'],
  'trim': ['Burn-in trim', 'boolean'],
  'outliers': ['Outliers', ["none", "iqr", "grubbs", "mahal"]]
}

const OPTIONS_HELP = {
  'dream': 'if Steps=0, Steps will be calculated as (Burn-in steps) + (Samples / (Population * Num. Fit Params))'
}

// make another working copy for editing:
// const active_settings = ref<{ name: string, settings: object}>({name: "", settings: {}});
const active_settings = ref({});

props.socket.asyncEmit("get_fitter_defaults", (new_fitter_defaults) => {
  console.log({new_fitter_defaults});
  fitter_defaults.value = new_fitter_defaults;
  fitter_settings.value = structuredClone(new_fitter_defaults);
})

props.socket.on('fitter_settings', (new_fitter_settings) => {
  fitter_settings.value = new_fitter_settings;
});

props.socket.on('selected_fitter', (new_selected_fitter: string) => {
  selected_fitter.value = new_selected_fitter;
});

let modal: Modal;

onMounted(() => {
  modal = new Modal(dialog.value, { backdrop: 'static', keyboard: false });

});

function close() {
  modal?.hide();
}

function open() {
  // copy the  selected_fitter_local from the server state:
  selected_fitter_local.value = selected_fitter.value;
  changeActiveFitter();
  modal?.show();
}

const fit_names = computed(() => Object.keys(fitter_defaults?.value));

function changeActiveFitter() {
  active_settings.value = structuredClone(fitter_settings.value[selected_fitter_local.value]?.settings) ?? {};
}

function process_settings() {
  return Object.fromEntries(Object.entries(active_settings.value).map(([sname, value]) => {
    const field_type = FIT_FIELDS[sname][1];
    let processed_value: any = value;
    if (field_type === 'integer') {
      processed_value = Math.round(Number(value));
    }
    else if (field_type === 'float') {
      processed_value = Number(value);
    }
    else if (field_type === 'boolean') {
      // probably unnecessary if it is bound to a checkbox
      processed_value = Boolean(value);
    }
    return [sname, processed_value];
  }))
}

async function save(start: boolean = false) {
  if (anyIsInvalid.value) {
    return;
  }
  const new_settings = structuredClone(fitter_settings.value);
  new_settings[selected_fitter_local.value] = { settings: process_settings() };
  const fitter_settings_local = process_settings();
  new_settings[selected_fitter_local.value] = { settings: fitter_settings_local };
  await props.socket.asyncEmit("set_shared_setting", "fitter_settings", new_settings);
  await props.socket.asyncEmit("set_shared_setting", "selected_fitter", selected_fitter_local.value);
  if (start) {
    await props.socket.asyncEmit("start_fit_thread", selected_fitter_local.value, fitter_settings_local);
  }
  close();
}

function reset() {
  active_settings.value = structuredClone(fitter_defaults.value[selected_fitter_local.value].settings) ?? {};
}

function validate(value, field_name) {
  const field_type = FIT_FIELDS[field_name][1];
  if (Array.isArray(field_type) || field_type === 'boolean') {
    // there's no way to get an incorrect option.
    return true;
  }
  const float_value = Number(value);
  if (isNaN(float_value)) {
    return false;
  }
  if (field_type === 'integer' && parseInt(value, 10) != float_value) {
    return false;
  }
  return true;
}

const anyIsInvalid = computed(() => {
  return Object.entries(active_settings.value).some(([sname, value]) => !validate(value, sname));
})

onMounted(async () => {
  const new_selected_fitter = await props.socket.asyncEmit('get_shared_setting', 'selected_fitter');
  const new_fitter_settings = await props.socket.asyncEmit('get_shared_setting', 'fitter_settings');
  if (new_selected_fitter !== undefined) {
    selected_fitter.value = new_selected_fitter;
  }
  if (new_fitter_settings !== undefined) {
    fitter_settings.value = new_fitter_settings;
  }
});

defineExpose({
  close,
  open,
})
</script>

<template>
  <div ref="dialog" class="modal fade" id="fitOptionsModal" tabindex="-1" aria-labelledby="fitOptionsLabel"
    :aria-hidden="isOpen">
    <div class="modal-dialog modal-lg">
      <div class="modal-content">
        <div class="modal-header">
          <h5 class="modal-title" id="fitOptionsLabel">Fit Options</h5>
          <button type="button" class="btn-close" @click="close" aria-label="Close"></button>
        </div>
        <div class="modal-body">
          <div class="container">
            <div class="row border-bottom">
              <div class="col">
                <div class="form-check" v-for="fname in fit_names.slice(0,3)" :key="fname">
                  <input class="form-check-input" v-model="selected_fitter_local" type="radio" name="flexRadio"
                    :id="fname" :value="fname" @change="changeActiveFitter">
                  <label class="form-check-label" :for="fname">
                    {{fitter_defaults[fname].name}}
                    <span v-if="fname !== 'scipy.leastsq'">({{ fname }})</span>
                  </label>
                </div>
              </div>
              <div class="col">
                <div class="form-check" v-for="fname in fit_names.slice(3)" :key="fname">
                  <input class="form-check-input" v-model="selected_fitter_local" type="radio" name="flexRadio"
                    :id="fname" :value="fname" @change="changeActiveFitter">
                  <label class="form-check-label" :for="fname">
                    {{fitter_defaults[fname].name}}
                    <span v-if="fname !== 'scipy.leastsq'">({{ fname }})</span>
                  </label>
                </div>
              </div>
            </div>
            <div class="row p-2">
              <div class="row p-1 text-secondary" v-if="selected_fitter_local in OPTIONS_HELP">
                <span><em>{{ OPTIONS_HELP[selected_fitter_local] }}</em></span>
              </div>
              <div class="row p-1" v-for="(value, sname, index) in active_settings" :key="sname">
                <label class="col-sm-4 col-form-label" :for="'setting_' + index">{{FIT_FIELDS[sname][0]}}</label>
                <div class="col-sm-8">
                  <select v-if="Array.isArray(FIT_FIELDS[sname][1])" v-model="active_settings[sname]" class="form-select">
                    <option v-for="opt in FIT_FIELDS[sname][1]">{{opt}}</option>
                  </select>
                  <input v-else-if="FIT_FIELDS[sname][1]==='boolean'" class="form-check-input m-2" type="checkbox"
                    v-model="active_settings[sname]" />
                  <input v-else :class="{'form-control': true, 'is-invalid': !validate(active_settings[sname], sname)}"
                    type="text" v-model="active_settings[sname]" @keydown.enter="save" />
                </div>
              </div>
            </div>
          </div>
        </div>
        <div class="modal-footer">
          <button type="button" class="btn btn-secondary" @click="reset">Reset Defaults</button>
          <button type="button" class="btn btn-success" :class="{disabled: anyIsInvalid}" @click="save(true)">Save and Start</button>
          <button type="button" class="btn btn-primary" :class="{disabled: anyIsInvalid}" @click="save(false)">
            Save Changes</button>

        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.form-check label {
  user-select: none;
}
</style>