<script setup lang="ts">
import { ref, onMounted } from 'vue';
import type { Socket } from 'socket.io-client';

// type PanelComponent = DefineComponent<{socket: Socket, visible: boolean}, any>;

const tabTriggers = ref<HTMLAnchorElement[]>([]);

const props = defineProps<{
  socket: Socket,
  panels: {title: string, component: unknown}[],
  active_panel: number
}>();

const emit = defineEmits<{
  (e: 'panel_changed', active_panel: number): void
}>();


function setActive(index: number) {
  emit("panel_changed", index);
}

</script>

<template>
  <ul class="nav nav-tabs">
    <li class="nav-item" v-for="(panel, index) in props.panels" :key="index">
      <a ref="tabTriggers" :class="{'nav-link': true, active: index == active_panel}" href="#" @click="setActive(index)">{{panels[index]?.title}}</a>
    </li>
  </ul>
  <div class="tab-content d-flex flex-column flex-grow-1 overflow-auto">
      <component :is="panels[active_panel].component" :socket="props.socket" :visible="true"></component>
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