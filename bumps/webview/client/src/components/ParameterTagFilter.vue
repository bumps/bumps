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
  // const tag_names = Array.from(new Set(props.parameters.map((p) => p.tags ?? []).flat()));
  // const tag_names = ["tag1", "tag2", "tag3", "tag4", "tag5", "tag6", "tag7", "tag8", "tag9", "tag10"];
  const tag_names = Array.from(new Set(props.parameters.map((p) => p.name).flat()));
  return Object.fromEntries(tag_names.map((t, i) => [t, COLORS[i % COLORS.length]]));
});

const COLORS = [
  "#af0e2b" /* red */,
  "#be460f" /* orange */,
  "#be9500" /* yellow */,
  "#136d01" /* green */,
  "#0b6e6e" /* teal */,
  "#0b3e6e" /* blue */,
  "#6A07B6" /* purple */,
  "#a70a9d" /* pink */,
];

const getBackgroundColor = (tag: string, color: string, toHide) => {
  let bgColor: string;
  if (toHide) {
    bgColor = tags_to_show.value.includes(tag as string) ? `${color}FF` : `${color}66`;
  } else {
    bgColor = tags_to_hide.value.includes(tag as string) ? `${color}FF` : `${color}66`;
  }
  return bgColor;
};

function should_show(tags: string[]) {
  if (tags_to_hide.value.length > 0 && tags.some((t) => tags_to_hide.value.includes(t))) {
    return false;
  }
  // then we're not specifically hiding it...
  else if (tags_to_show.value.length > 0) {
    if (tags.some((t) => tags_to_show.value.includes(t))) {
      return true;
    } else {
      return false;
    }
  }
  // then we're not specifying to_show, show by default:
  else {
    return true;
  }
}

const filtered_parameters = computed(() => {
  // return list of { parameter, index } objects
  // that should be shown based on tag filters
  return props.parameters
    .map((parameter, index) => {
      return {
        parameter,
        index,
        show: should_show(parameter.tags),
      };
    })
    .filter(({ show }) => show);
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
          :key="`include-tag-${tag}`"
          class="badge rounded-pill me-1"
          :style="{
            color: 'white',
            'background-color': `${getBackgroundColor(tag, tag_color, true)}`,
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
            'background-color': `${getBackgroundColor(tag, tag_color, false)}`,
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
