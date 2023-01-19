<script setup lang="ts">
import { ref, onMounted, watch, onUpdated, computed, shallowRef } from 'vue';
import type { DefineComponent, Component } from 'vue';
import type { Socket } from 'socket.io-client';

// type PanelComponent = DefineComponent<{socket: Socket, visible: boolean}, any>;

const tabTriggers = ref<HTMLAnchorElement[]>([]);
const panelContainers = ref<HTMLDivElement[]>([]);
const activePanel = ref(0);

const props = defineProps<{
  socket: Socket,
  panels: {title: string, component: unknown}[],
  initially_active?: number
}>();

onMounted(() => {
  activePanel.value = props.initially_active ?? 0;
});
</script>

<template>
  <ul class="nav nav-tabs">
    <li class="nav-item" v-for="(panel, index) in props.panels" :key="index">
      <a ref="tabTriggers" :class="{'nav-link': true, active: index == activePanel}" href="#" @click="activePanel = index">{{panels[index]?.title}}</a>
    </li>
  </ul>
  <div class="tab-content d-flex flex-column flex-grow-1 overflow-auto">
      <component :is="panels[activePanel].component" :socket="props.socket" :visible="true"></component>
  </div>
</template>

<style scoped>
.tab-content>.tab-pane {
  display: none;
}

.tab-content>.tab-pane.active {
  display: flex !important;
}
</style>