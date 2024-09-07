<script setup lang="ts">
import { ref, onMounted } from 'vue';
import type { AsyncSocket } from '../asyncSocket.ts';

// type PanelComponent = DefineComponent<{socket: Socket, visible: boolean}, any>;

const tabTriggers = ref<HTMLAnchorElement[]>([]);

const props = defineProps<{
  socket: AsyncSocket,
  panels: {title: string, component: unknown}[],
  active_panel: number,
  hide_tabs?: boolean,
}>();

const emit = defineEmits<{
  (e: 'panel_changed', active_panel: number): void
}>();


function setActive(index: number) {
  emit("panel_changed", index);
}

</script>

<template>
  <ul class="nav nav-tabs" v-if="!hide_tabs">
    <li class="nav-item" v-for="(panel, index) in props.panels" :key="index">
      <a ref="tabTriggers" :class="{'nav-link': true, active: index == active_panel}" href="#" @click="setActive(index)">{{panels[index]?.title}}</a>
    </li>
  </ul>
  <div class="tab-content d-flex flex-column flex-grow-1 overflow-auto">
    <KeepAlive>
      <component :is="panels[active_panel].component" :socket="props.socket"></component>
    </KeepAlive>
  </div>
</template>

<style scoped>
.tab-content>.tab-pane {
  display: none;
}

.tab-content {
  display: flex !important;
  flex-grow: 1;
  flex-shrink: 0;
  flex-basis: 200px;
}
</style>