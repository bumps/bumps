<script setup lang="ts">
import { computed, ref, watch } from "vue";

const props = defineProps<{
  name: string | number;
  value: any;
  parentObj?: any; // <-- ADD THIS to track the parent proxy
  references?: Record<string, any>;
  path?: string;
  visiblePaths?: Set<string> | null;
  expandPaths?: Set<string> | null;
  collapseTrigger?: number;
  startExpanded?: boolean;
}>();

// ... (keep all your existing computed properties: currentPath, isVisible, actualValue, etc.) ...
const currentPath = computed(() => (props.path ? `${props.path}.${props.name}` : String(props.name)));
const isVisible = computed(() => !props.visiblePaths || props.visiblePaths.has(currentPath.value));
const isOpen = ref(props.startExpanded ?? false);

watch(
  () => props.collapseTrigger,
  () => (isOpen.value = false),
);
watch(
  () => props.expandPaths,
  (newExpand) => {
    if (newExpand) isOpen.value = newExpand.has(currentPath.value);
  },
);

const isRef = computed(() => props.value && typeof props.value === "object" && props.value.__class__ === "Reference");

const actualValue = computed(() => {
  if (isRef.value && props.references && props.value.id) return props.references[props.value.id] ?? props.value;
  return props.value;
});

const nodeClass = computed(() => actualValue.value?.__class__ || "");
const isBumpsParameter = computed(() => nodeClass.value === "bumps.parameter.Parameter");
const isDynamicObject = computed(() => nodeClass.value && !isBumpsParameter.value);

const isPrimitive = computed(() => actualValue.value === null || typeof actualValue.value !== "object");
const isArray = computed(() => Array.isArray(actualValue.value));
const isPrimitiveArray = computed(
  () => isArray.value && actualValue.value.every((v: any) => v === null || typeof v !== "object"),
);

const summaryText = computed(() => {
  const val = actualValue.value;
  if (isArray.value) return `Array(${val.length})`;
  const cls = val.__class__ ? val.__class__.split(".").pop() : "";
  const name = val.name ? `"${val.name}"` : "";
  return [cls, name].filter(Boolean).join(" ") || "Object";
});

const primitiveArrayPreview = computed(() => {
  if (!isPrimitiveArray.value) return "";
  const val = actualValue.value;
  return val.length <= 10 ? `[${val.join(", ")}]` : `[${val.slice(0, 10).join(", ")}, ... (${val.length - 10} more)]`;
});

const onToggle = (event: Event) => (isOpen.value = (event.target as HTMLDetailsElement).open);
</script>

<template>
  <div class="model-node" v-if="isVisible">

    <div v-if="isBumpsParameter" class="inline-editor">
      <span class="key" :title="nodeClass">{{ name }}:</span>
      <span v-if="isRef" class="ref-badge" title="Dereferenced">⤤</span>

      <div class="param-controls">
        <span class="param-name" v-if="actualValue.name">"{{ actualValue.name }}"</span>

        <label class="control-group">
          Value:
          <input
            type="number"
            class="num-input"
            v-if="actualValue.slot && actualValue.slot.__class__ === 'bumps.parameter.Variable'"
            v-model.number="actualValue.slot.value"
          />
          <span class="fallback-val" v-else>{{ actualValue.slot }}</span>
        </label>

        <label class="control-group checkbox-group">
          <input type="checkbox" v-model="actualValue.fixed" /> Fixed
        </label>
      </div>
    </div>

    <template v-else-if="isPrimitive">
      <span class="key">{{ name }}:</span>
      <input v-if="typeof actualValue === 'number'" type="number" class="primitive-input" v-model.number="parentObj[name]" />
      <input v-else-if="typeof actualValue === 'boolean'" type="checkbox" v-model="parentObj[name]" />
      <input v-else type="text" class="primitive-input" v-model="parentObj[name]" />
    </template>

    <details v-else-if="isPrimitiveArray" :open="isOpen" @toggle="onToggle">
      <summary>
        <span class="key">{{ name }}:</span>
        <span class="val array-preview">{{ primitiveArrayPreview }}</span>
      </summary>
      <div class="children full-array">[{{ actualValue.join(", ") }}]</div>
    </details>

    <details v-else :open="isOpen" @toggle="onToggle">
      <summary>
        <span class="key">{{ name }}</span>
        <span v-if="isRef" class="ref-badge" title="Dereferenced Parameter">⤤</span>
        <span class="summary-text">{{ summaryText }}</span>
      </summary>
      <div class="children">
        <template v-for="(childVal, childKey) in actualValue" :key="childKey">
          <ModelNode
            v-if="childKey !== '__class__' && childKey !== 'id'"
            :name="childKey"
            :value="childVal"
            :parentObj="actualValue"
            :references="references"
            :path="currentPath"
            :visible-paths="visiblePaths"
            :expand-paths="expandPaths"
            :collapse-trigger="collapseTrigger"
          />
        </template>
      </div>
    </details>

  </div>
</template>

<style scoped>
.model-node {
  font-family: monospace;
  font-size: 13px;
  line-height: 1.5;
  margin: 2px 0;
}
.children { margin-left: 1.5rem; border-left: 1px solid #e0e0e0; padding-left: 0.5rem; }
.full-array { color: #444; word-break: break-all; white-space: normal; }

.key { font-weight: bold; color: #881391; margin-right: 4px; }
.val.array-preview { color: #555; font-style: italic; }

summary { cursor: pointer; user-select: none; display: list-item; }
summary:hover { background-color: #f5f5f5; border-radius: 4px; }
.summary-text { color: #666; margin-left: 0.5rem; }
.ref-badge { color: #d94b2b; font-weight: bold; margin-left: 0.3rem; margin-right: 0.3rem; }
details > summary::marker { color: #aaa; }

/* Editor Styles */
.inline-editor {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  background-color: #fdfdfd;
  border: 1px solid #eee;
  padding: 2px 6px;
  border-radius: 4px;
}
.param-controls {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-left: 8px;
}
.param-name { color: #1a1aa6; font-style: italic; }
.control-group { display: flex; align-items: center; gap: 4px; color: #555; }
.checkbox-group { cursor: pointer; }
.num-input {
  width: 70px;
  padding: 2px 4px;
  font-family: monospace;
  font-size: 12px;
  border: 1px solid #ccc;
  border-radius: 3px;
}
.primitive-input {
  border: 1px dashed #ccc;
  border-radius: 2px;
  padding: 1px 4px;
  font-family: monospace;
  color: #1a1aa6;
  background: transparent;
}
.primitive-input:focus { border-style: solid; border-color: #007acc; outline: none; background: white;}
</style>
