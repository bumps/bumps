<script setup lang="ts">
import { ref, watch } from "vue";
import type { AsyncSocket } from "../asyncSocket.ts";

// type PanelComponent = DefineComponent<{socket: Socket, visible: boolean}, any>;

const tabTriggers = ref<HTMLAnchorElement[]>([]);

const props = defineProps<{
  socket: AsyncSocket;
  panels: { title: string; component: unknown; show?: () => boolean }[];
  startPanel: number;
  hideTabs?: boolean;
}>();

const activePanel = ref(props.startPanel);

function setActive(index: number) {
  activePanel.value = index;
}

// watch for changes to panel.show() and update the tabs
watch(
  () => props.panels.map((p) => p.show?.()),
  (newValue) => {
    if (newValue[activePanel.value] === false) {
      // revert to startPanel if the active panel is hidden
      activePanel.value = props.startPanel;
    }
  }
);
</script>

<template>
  <ul v-if="!hideTabs" class="nav nav-tabs">
    <li v-for="(panel, index) in props.panels" :key="index" class="nav-item">
      <a
        v-if="panel?.show?.() ?? true"
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
