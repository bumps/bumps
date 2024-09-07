<script setup lang="ts">
import { ref, computed } from 'vue';

type parameter_info = {
  tags: string[]
}

const props = defineProps<{parameters: parameter_info[]}>();

const tags_to_show = ref<string[]>([]);
const tags_to_hide = ref<string[]>([]);
const show_tags = ref(false);

function toggle(tag: string, listname: 'show' | 'hide') {
  const list = (listname === 'show') ? tags_to_show.value : tags_to_hide.value;
  const tag_index = list.indexOf(tag);
  if (tag_index > -1) {
    list.splice(tag_index, 1);
  }
  else {
    list.push(tag);
  }
}

const tag_colors = computed(() => {
  const tag_names = Array.from(new Set(props.parameters.map((p) => p.tags ?? []).flat()));
  return Object.fromEntries(tag_names.map((t,i) => [t, COLORS[i%COLORS.length]]));
});

const COLORS = [
  "blue",
  "red",
  "green",
  "goldenrod",
  "grey",
  "orange",
  "purple",
  "teal",
  "lightgreen",
  "brown",
  "black"
];

const filtered_parameters = computed(() => {
  return props.parameters.filter(({tags}: {tags: string[]}) => {
    if ((tags_to_hide.value.length > 0) && tags.some((t) => tags_to_hide.value.includes(t))) {
      return false;
    }
    // then we're not specifically hiding it...
    else if (tags_to_show.value.length > 0) {
      if (tags.some((t) => tags_to_show.value.includes(t))) {
        return true;
      }
      else {
        return false;
      }
    }
    // then we're not specifying to_show, show by default:
    else {
      return true;
    }
  });
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
    <summary>tags</summary>
    <div class="form-check">
      <label class="form-check-label ps-1">
        <input type="checkbox" class="form-check-input" v-model="show_tags" />
        display tags
      </label>
    </div>
    <h6>Filters:</h6>
    <div class="row pb-1 ps-1">
      <div class="col-1 text-end">include</div>
      <div class="col">
      <span 
        class="badge rounded-pill me-1" 
        :class="{checked: tags_to_show.includes(tag)}" 
        v-for="(tag_color, tag) in tag_colors" 
        @click="toggle(tag, 'show')"
        :style="{color: 'white', 'background-color': tag_color}"
        >
        {{ tag }}
      </span>
      </div>
    </div>
    <div class="row pb-1 ps-1">
      <div class="col-1 text-end">exclude</div>
      <div class="col">
        <span 
        class="badge rounded-pill me-1" 
        :class="{checked: tags_to_hide.includes(tag)}" 
        v-for="(tag_color, tag, tag_index) in tag_colors" 
        @click="toggle(tag, 'hide')"
        :style="{color: 'white', 'background-color': tag_color}"
        >
        {{ tag }}
      </span>
      </div>
    </div>
  </details>
</template>

<style scoped>
.filters {
  user-select: none;
  display: inline-block;
}
span.badge.checked {
  opacity: 1;
}

span.badge {
  opacity: 0.3;
  cursor: pointer;
}
</style>