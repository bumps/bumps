<script setup lang="ts">
import { nextTick, ref } from 'vue';

const props = defineProps<{
  title: string,
}>();

const expanded = ref(false);
const dropdown = ref<HTMLElement>();

function check_for_click_outside(ev) {
  // Check if the target of the mousedown is inside the dropdown
  // menu. If not, then hide the menu on mouseup.
  if (!(dropdown.value?.contains?.(ev.target) ?? false)) {
    document.addEventListener('mouseup', hide, { once: true });
    document.removeEventListener('mousedown', check_for_click_outside);
  }
}

function show() {
  expanded.value = true;
  window.addEventListener('mousedown', check_for_click_outside);
}

function hide() {
  window.removeEventListener('mousedown', check_for_click_outside);
  expanded.value = false;
}

function toggle_show() {
  if (expanded.value) {
    hide();
  } else {
    show();
  }
}


</script>

<template>
  <li ref="dropdown" class="nav-item dropdown">
    <button class="btn btn-link nav-link dropdown-toggle" role="button"
      @click="toggle_show"
      :aria-expanded="expanded">
      {{ props.title }}
    </button>
    <ul :class="{show: expanded, 'dropdown-menu': true}">
      <slot :hide="hide"></slot>
    </ul>
  </li>
</template>