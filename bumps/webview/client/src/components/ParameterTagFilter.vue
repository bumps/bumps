<script setup lang="ts">
import { computed, ref } from "vue";

type parameter_info = {
  name: string;
  tags: string[];
};

const props = defineProps<{ parameters: parameter_info[] }>();

const tags_to_show = ref<string[]>([]);
const tags_to_hide = ref<string[]>([]);
const show_tags = ref(false);

function toggle(tag: string, listname: "show" | "hide") {
  const list = listname === "show" ? tags_to_show.value : tags_to_hide.value;
  const tag_index = list.indexOf(tag);
  if (tag_index > -1) {
    list.splice(tag_index, 1);
  } else {
    list.push(tag);
  }
}

const tag_colors = computed(() => {
  // const tag_names = Array.from(new Set(props.parameters.map((p) => p.name).flat())); // test with all parameters
  const tag_names = Array.from(new Set(props.parameters.map((p) => p.tags ?? []).flat()));
  let tagColors = Object.fromEntries(tag_names.map((t, i) => [t, COLORS[i % COLORS.length]]));
  return tagColors;
});

const COLORS: string[] = [
  "#af0e2b" /** red */,
  "#be460f" /** orange */,
  "#be9500" /** yellow */,
  "#136d01" /** green */,
  "#0b6e6e" /** teal */,
  "#0b3e6e" /** blue */,
  "#6A07B6" /** purple */,
  "#a70a9d" /** pink */,
];

const getBackgroundColor = (tag: string, color: string, toHide: boolean) => {
  let bgColor: string;
  if (toHide) {
    bgColor = tags_to_show.value.includes(tag) ? `${color}FF` : `${color}66`;
  } else {
    bgColor = tags_to_hide.value.includes(tag) ? `${color}FF` : `${color}66`;
  }
  return bgColor;
};

function shouldShow(tags: string[]): boolean {
  // if we're specifically hiding it, then don't show it
  if (tags_to_hide.value.length > 0 && tags.some((t) => tags_to_hide.value.includes(t))) {
    return false;
  }
  if (tags_to_show.value.length > 0) {
    // true if tag is in tags_to_show, false otherwise
    return tags.some((t) => tags_to_show.value.includes(t));
  }
  // show by default
  return true;
}

const filtered_parameters = computed(() => {
  // return list of { parameter, index } objects
  // that should be shown based on tag filters
  let filtered = props.parameters
    .map((parameter, index) => {
      return {
        parameter,
        index,
        show: shouldShow(parameter.tags),
      };
    })
    .filter(({ show }) => show);

  return filtered;
});

defineExpose({
  tags_to_show,
  tags_to_hide,
  show_tags,
  tag_colors,
  filtered_parameters,
});
</script>

<template>
  <details ref="tag_filters" class="filters">
    <summary>Tags</summary>
    <div class="form-check">
      <label class="form-check-label ps-1">
        <input v-model="show_tags" type="checkbox" class="form-check-input" />
        Display tags
      </label>
    </div>
    <h6>Filters:</h6>
    <div class="row pb-1 ps-1">
      <div class="col-1 text-end">Include</div>
      <div class="col">
        <button
          v-for="(tag_color, tag) in tag_colors"
          :id="`include-tag-${tag}`"
          :key="`include-tag-${tag}`"
          class="badge rounded-pill me-1"
          :style="{
            color: 'white',
            'background-color': `${getBackgroundColor(tag as string, tag_color, true)}`,
          }"
          @click="toggle(tag as string, 'show')"
          @keydown.enter="toggle(tag as string, 'show')"
        >
          {{ tag }}
        </button>
      </div>
    </div>
    <div class="row pb-1 ps-1">
      <div class="col-1 text-end">Exclude</div>
      <div class="col">
        <button
          v-for="(tag_color, tag) in tag_colors"
          :key="`exclude-tag-${tag}`"
          class="badge rounded-pill me-1"
          :style="{
            color: 'white',
            'background-color': `${getBackgroundColor(tag as string, tag_color, false)}`,
          }"
          @click="toggle(tag as string, 'hide')"
          @keydown.enter="toggle(tag as string, 'hide')"
        >
          {{ tag }}
        </button>
      </div>
    </div>
  </details>
</template>

<style scoped>
.filters {
  display: inline-block;
  user-select: none;
}

.badge {
  border: 0;
}

/* span.badge.checked {
  opacity: 1;
}

span.badge {
  cursor: pointer;
  opacity: 0.4;
} */
</style>
