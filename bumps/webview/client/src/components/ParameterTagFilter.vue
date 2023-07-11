<script setup lang="ts">
import { ref, computed } from 'vue';

const props = defineProps<{all_tags: {[tag: string]: string}}>();

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

defineExpose({
  tags_to_show,
  tags_to_hide,
  show_tags,
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
        v-for="(tag_color, tag) in all_tags" 
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
        v-for="(tag_color, tag) in all_tags" 
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