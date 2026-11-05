<script setup lang="ts">
import { ref, watch } from "vue";
import type { AsyncSocket } from "../asyncSocket.ts";

// type PanelComponent = DefineComponent<{socket: Socket, visible: boolean}, any>;

const _tabTriggers = ref<HTMLButtonElement[]>([]);

const props = defineProps<{
  socket: AsyncSocket;
  panels: { title: string; component: unknown; show?: () => boolean }[];
  defaultPanel: number;
  hideTabs?: boolean;
}>();

// bound to app state so the selection survives this container being recreated
const activePanel = defineModel<number>("activePanel", { required: true });

function setActive(index: number) {
  activePanel.value = index;
}

// watch for changes to panel.show() and update the tabs
// (immediate, since the remembered panel may have been hidden while this container was unmounted)
watch(
  () => props.panels.map((p) => p.show?.()),
  (newValue) => {
    if (newValue[activePanel.value] === false) {
      // revert to defaultPanel if the active panel is hidden
      activePanel.value = props.defaultPanel;
    }
  },
  { immediate: true }
);
</script>

<template>
  <div v-if="!hideTabs" class="nav nav-tabs" role="tablist">
    <template v-for="(panel, index) in props.panels" :key="index">
      <button
        v-if="panel?.show?.() ?? true"
        ref="tabTriggers"
        type="button"
        role="tab"
        :class="{ 'nav-link': true, active: index === activePanel }"
        :aria-selected="index === activePanel"
        @click="setActive(index)"
      >
        {{ panels[index]?.title }}
      </button>
    </template>
  </div>
  <div class="tab-content d-flex flex-column flex-grow-1 overflow-auto">
    <KeepAlive>
      <component :is="panels[activePanel]?.component" :socket="props.socket"></component>
    </KeepAlive>
  </div>
</template>

<style scoped>
.tab-content > .tab-pane {
  display: none;
}

.tab-content {
  display: flex;
  flex-grow: 1;
  flex-shrink: 0;
  flex-basis: 200px;
}
</style>
