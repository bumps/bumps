<script setup lang="ts">
import { onMounted, ref } from "vue";
import type { AsyncSocket } from "../asyncSocket.ts";

// type PanelComponent = DefineComponent<{socket: Socket, visible: boolean}, any>;

const tabTriggers = ref<HTMLAnchorElement[]>([]);

const props = defineProps<{
  socket: AsyncSocket;
  panels: { title: string; component: unknown }[];
  activePanel: number;
  hideTabs?: boolean;
}>();

const emit = defineEmits<{
  (e: "panel_changed", activePanel: number): void;
}>();

function setActive(index: number) {
  emit("panel_changed", index);
}
</script>

<template>
  <ul v-if="!hideTabs" class="nav nav-tabs">
    <li v-for="(panel, index) in props.panels" :key="index" class="nav-item">
      <a
        ref="tabTriggers"
        :class="{ 'nav-link': true, active: index == activePanel }"
        href="#"
        @click.prevent="setActive(index)"
        >{{ panels[index]?.title }}</a
      >
    </li>
  </ul>
  <div class="tab-content d-flex flex-column flex-grow-1 overflow-auto">
    <KeepAlive>
      <component :is="panels[activePanel].component" :socket="props.socket"></component>
    </KeepAlive>
  </div>
</template>

<style scoped>
.tab-content > .tab-pane {
  display: none;
}

.tab-content {
  display: flex !important;
  flex-grow: 1;
  flex-shrink: 0;
  flex-basis: 200px;
}
</style>
