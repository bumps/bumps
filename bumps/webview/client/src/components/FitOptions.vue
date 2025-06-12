<script setup lang="ts">
import { computed, onMounted, ref, shallowRef, toRaw } from "vue";
import { default_fitter, default_fitter_settings, shared_state } from "../app_state";
import type { AsyncSocket } from "../asyncSocket";

const props = defineProps<{ socket: AsyncSocket }>();

const dialog = ref<HTMLDialogElement>();
const isOpen = ref(false);
const selected_fitter_local = ref("amoeba");

interface FitField {
  name: string;
  label: string;
  stype: "integer" | "float" | "boolean" | { min: number; max: number } | string[];
  description: string;
  fitters: string[];
  defaults: any[];
}

const fit_fields = shallowRef<{ [key: string]: FitField }>({});

const OPTIONS_HELP: { [key: string]: string } = {
  dream: "if Steps=0, Steps will be calculated as (Burn-in steps) + (Samples / (Population * Num. Fit Params))",
};

// make another working copy for editing:
// const active_settings = ref<{ name: string, settings: object}>({name: "", settings: {}});
const active_settings = ref({});

function close() {
  isOpen.value = false;
  dialog.value?.close();
}

function open() {
  // copy the  selected_fitter_local from the server state:
  selected_fitter_local.value = shared_state.selected_fitter ?? default_fitter;
  changeActiveFitter();
  isOpen.value = true;
  dialog.value?.showModal();
}

const fit_names = computed(() => Object.keys(default_fitter_settings.value));
const fitter_settings_with_defaults = computed(() => {
  return shared_state.fitter_settings ?? default_fitter_settings.value;
});

function changeActiveFitter() {
  const fs = toRaw(fitter_settings_with_defaults.value);
  const cloned = structuredClone({ ...fs[selected_fitter_local.value]?.settings }) ?? {};
  // reject stored fit options that aren't defined in FIT_FIELDS above
  const new_settings = Object.fromEntries(Object.entries(cloned).filter(([k]) => k in fit_fields.value));
  active_settings.value = new_settings;
}

function process_settings() {
  return Object.fromEntries(
    Object.entries(active_settings.value).map(([sname, value]) => {
      const field_type = fit_fields.value[sname].stype;
      let processed_value: any = value;
      if (field_type === "integer") {
        processed_value = Math.round(Number(value));
      } else if (field_type === "float") {
        processed_value = Number(value);
      } else if (field_type === "boolean") {
        // probably unnecessary if it is bound to a checkbox
        processed_value = Boolean(value);
      } else if (Array.isArray(field_type)) {
        // this is a list of options
        processed_value = value;
      } else if (typeof field_type === "object") {
        // this is a range
        // Check for Array.isArray(field_type) first, because
        // typeof field_type === "object" will also be true for arrays
        processed_value = Number(value);
      }
      return [sname, processed_value];
    })
  );
}

async function save(start: boolean = false, resume: boolean = false) {
  if (anyIsInvalid.value) {
    return;
  }
  const new_settings = structuredClone({ ...toRaw(fitter_settings_with_defaults.value) });
  const name = selected_fitter_local.value;
  const fitter_settings_local = process_settings();
  new_settings[name] = { name, settings: fitter_settings_local };
  await props.socket.asyncEmit("set_shared_setting", "fitter_settings", new_settings);
  await props.socket.asyncEmit("set_shared_setting", "selected_fitter", selected_fitter_local.value);
  if (start) {
    // start_fit_thread(
    //   fitter_name: string,
    //   fitter_settings: object,
    //   // max_time: float = 0.0, // removed again from the server
    //   resume: boolean = false
    // )
    await props.socket.asyncEmit("start_fit_thread", selected_fitter_local.value, fitter_settings_local, resume);
  }
  close();
}

function reset() {
  active_settings.value =
    structuredClone({ ...default_fitter_settings.value[selected_fitter_local.value].settings }) ?? {};
}

function validation_error(value: any, field_name: string) {
  const field_type = fit_fields.value[field_name].stype;
  if (Array.isArray(field_type) || field_type === "boolean") {
    // there's no way to get an incorrect option.
    return null;
  }
  const float_value = Number(value);
  if (isNaN(float_value)) {
    return "not a number";
  }
  if (field_type === "integer" && parseInt(value, 10) != float_value) {
    return "not an integer";
  } else if (typeof field_type === "object") {
    if ("min" in field_type && float_value < field_type.min) {
      return `must be >= ${field_type.min}`;
    }
    if ("max" in field_type && float_value > field_type.max) {
      return `must be <= ${field_type.max}`;
    }
  }
  return null;
}

const anyIsInvalid = computed(() => {
  return Object.entries(active_settings.value).some(([sname, value]) => validation_error(value, sname) !== null);
});

onMounted(async () => {
  const server_fit_fields = await props.socket.asyncEmit("get_fit_fields");
  fit_fields.value = server_fit_fields;
});

defineExpose({
  close,
  open,
});
</script>

<template>
  <dialog ref="dialog">
    <div id="fitOptionsModal" class="modal show" tabindex="-1" aria-labelledby="fitOptionsLabel" :aria-hidden="!isOpen">
      <div class="modal-dialog modal-lg">
        <div class="modal-content">
          <div class="modal-header">
            <h5 id="fitOptionsLabel" class="modal-title">Fit Options</h5>
            <button type="button" class="btn-close" aria-label="Close" @click="close"></button>
          </div>
          <div class="modal-body">
            <div class="container">
              <div class="row border-bottom">
                <div class="col">
                  <div v-for="fname in fit_names.slice(0, 3)" :key="fname" class="form-check">
                    <input
                      :id="fname"
                      v-model="selected_fitter_local"
                      class="form-check-input"
                      type="radio"
                      name="flexRadio"
                      :value="fname"
                      @change="changeActiveFitter"
                    />
                    <label class="form-check-label" :for="fname">
                      {{ default_fitter_settings[fname].name }}
                      <span v-if="fname !== 'scipy.leastsq'">({{ fname }})</span>
                    </label>
                  </div>
                </div>
                <div class="col">
                  <div v-for="fname in fit_names.slice(3)" :key="fname" class="form-check">
                    <input
                      :id="fname"
                      v-model="selected_fitter_local"
                      class="form-check-input"
                      type="radio"
                      name="flexRadio"
                      :value="fname"
                      @change="changeActiveFitter"
                    />
                    <label class="form-check-label" :for="fname">
                      {{ default_fitter_settings[fname].name }}
                      <span v-if="fname !== 'scipy.leastsq'">({{ fname }})</span>
                    </label>
                  </div>
                </div>
              </div>
              <div class="row p-2">
                <div v-if="selected_fitter_local in OPTIONS_HELP" class="row p-1 text-secondary">
                  <span
                    ><em>{{ OPTIONS_HELP[selected_fitter_local] }}</em></span
                  >
                </div>
                <div v-for="(value, sname) in active_settings" :key="sname" class="row p-1">
                  <label
                    class="col-sm-4 col-form-label"
                    :for="`fitter_setting_${sname}`"
                    :title="fit_fields[sname].description"
                  >
                    {{ fit_fields[sname].label }}
                  </label>
                  <div class="col-sm-8">
                    <select
                      v-if="Array.isArray(fit_fields[sname].stype)"
                      :id="'fitter_setting_' + sname"
                      v-model="active_settings[sname]"
                      class="form-select"
                      :name="sname"
                    >
                      <option v-for="opt in fit_fields[sname].stype" :key="opt">
                        {{ opt }}
                      </option>
                    </select>
                    <input
                      v-else-if="fit_fields[sname].stype === 'boolean'"
                      :id="'fitter_setting_' + sname"
                      v-model="active_settings[sname]"
                      class="form-check-input m-2"
                      type="checkbox"
                      :name="sname"
                    />
                    <input
                      v-else
                      :id="'fitter_setting_' + sname"
                      v-model="active_settings[sname]"
                      :class="{
                        'form-control': true,
                        'is-invalid': validation_error(active_settings[sname], sname) !== null,
                      }"
                      type="text"
                      :name="sname"
                      :title="validation_error(active_settings[sname], sname) ?? fit_fields[sname].description"
                      @keydown.enter="() => save()"
                    />
                  </div>
                </div>
              </div>
            </div>
          </div>
          <div class="modal-footer">
            <button type="button" class="btn btn-secondary" @click="reset">Reset Defaults</button>
            <button type="button" class="btn btn-success" :class="{ disabled: anyIsInvalid }" @click="save(true)">
              Save and Start
            </button>
            <button type="button" class="btn btn-primary" :class="{ disabled: anyIsInvalid }" @click="save(false)">
              Save Changes
            </button>
            <button
              v-if="selected_fitter_local === shared_state.resumable"
              type="button"
              class="btn btn-warning"
              @click="save(true, true)"
            >
              Save and Resume
            </button>
          </div>
        </div>
      </div>
    </div>
  </dialog>
</template>

<style scoped>
.form-check label {
  user-select: none;
}

div.modal {
  display: block;
}
</style>
