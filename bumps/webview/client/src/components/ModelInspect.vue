<script setup lang="ts">
import { onMounted, ref, computed, toRaw } from "vue";
import { getDiff } from "json-difference";
import type { AsyncSocket } from "../asyncSocket.ts";
import { setupDrawLoop } from "../setupDrawLoop";
import ModelNode from "./ModelNode.vue";
import ParameterLinker from "./ParameterLinker.vue"; // Add this import

type json = string | number | boolean | null | json[] | { [key: string]: json };

const modelJson = ref<any>({});
const draftModel = ref<any>({}); // The local mutable copy
const searchQuery = ref("");
const collapseTrigger = ref(0);
const currentTab = ref("tree");

const props = defineProps<{
  socket: AsyncSocket;
}>();

setupDrawLoop("updated_parameters", props.socket, fetch_and_draw);

props.socket.on("model_loaded", () => {
  fetch_and_draw(true);
});

async function fetch_and_draw(reset: boolean = false) {
  const payload = (await props.socket.asyncEmit("get_model")) as string;
  const json_value: json = JSON.parse(payload);

  if (reset) {
    modelJson.value = json_value;
    draftModel.value = structuredClone(json_value); // Create the local draft
  } else {
    // If updating from server, you might want to handle merge conflicts here later.
    // For now, we update the original, and if the draft isn't dirty, update it too.
    const isClean = !isDirty.value;
    const old_model: json = modelJson.value as Record<string, any>;
    const new_model: json = json_value as Record<string, any>;
    const diff = getDiff(old_model, new_model);

    for (let [path, oldval, newval] of diff.edited) {
      const { target, parent, key } = resolve_diffpath(old_model, path);
      if (typeof key === "number" && Array.isArray(parent)) parent.splice(key, 1, newval);
      else parent[key] = newval;
    }

    if (isClean) {
      draftModel.value = structuredClone(modelJson.value);
    }
  }
}

onMounted(() => {
  fetch_and_draw(true);
});

const ARRAY_PATH_RE = /^([0-9]+)\[\]$/;
function resolve_diffpath(obj: Record<number | string, any>, diffpath: string) {
  const pathitems = diffpath.split("/");
  let target = obj;
  let parent: Record<number | string, any> = obj;
  let key: number | string = "";
  for (let pathitem of pathitems) {
    const array_path = pathitem.match(ARRAY_PATH_RE);
    parent = target;
    key = array_path ? Number(array_path[1]) : pathitem;
    target = target[key];
  }
  return { target, parent, key };
}

// Check if draft has been mutated
const isDirty = computed(() => {
  if (!modelJson.value || !draftModel.value) return false;
  console.log("isDirty?");
  // Use toRaw here to ensure clean comparison of primitives inside proxies
  return JSON.stringify(modelJson.value) !== JSON.stringify(draftModel.value);
});

const saveDraft = () => {
  // Push the draft back to the server
  props.socket.asyncEmit("set_serialized_problem", JSON.stringify(toRaw(draftModel.value)));
  // console.log("Saving model...", JSON.parse(JSON.stringify(draftModel.value)));
  // modelJson.value = JSON.parse(JSON.stringify(draftModel.value)); // Sync original to clear dirty state
  modelJson.value = structuredClone(toRaw(draftModel.value)); // Sync original to clear dirty state
};

const discardDraft = () => {
  draftModel.value = structuredClone(toRaw(modelJson.value));
};

// Search State Logic (same as previous)
const searchState = computed(() => {
  if (!searchQuery.value.trim() || !draftModel.value.object) {
    return { visible: null, expand: null };
  }

  const visiblePaths = new Set<string>();
  const expandPaths = new Set<string>();
  const query = searchQuery.value.trim().toLowerCase();
  const refs = draftModel.value.references || {};

  function traverse(obj: any, currentPath: string, ancestorMatched: boolean, key: string | number): boolean {
    let actualObj = obj;
    if (obj && typeof obj === "object" && obj.__class__ === "Reference" && refs[obj.id]) {
      actualObj = refs[obj.id];
    }

    let isMatch = false;
    if (ancestorMatched) {
      isMatch =
        String(key).toLowerCase().includes(query) ||
        (typeof actualObj !== "object" && actualObj !== null && String(actualObj).toLowerCase().includes(query));
    } else {
      isMatch =
        currentPath.toLowerCase().includes(query) ||
        (typeof actualObj !== "object" && actualObj !== null && String(actualObj).toLowerCase().includes(query));
    }

    let descendantMatched = false;
    const childAncestorMatched = ancestorMatched || isMatch;

    if (actualObj && typeof actualObj === "object") {
      for (const childKey in actualObj) {
        if (childKey === "__class__" || childKey === "id") continue;
        const childPath = currentPath ? `${currentPath}.${childKey}` : String(childKey);
        const childHasMatch = traverse(actualObj[childKey], childPath, childAncestorMatched, childKey);
        if (childHasMatch) descendantMatched = true;
      }
    }

    if (isMatch || descendantMatched || ancestorMatched) visiblePaths.add(currentPath);
    if (isMatch || descendantMatched) expandPaths.add(currentPath);

    return isMatch || descendantMatched;
  }

  traverse(draftModel.value.object, "FitProblem", false, "FitProblem");
  return { visible: visiblePaths, expand: expandPaths };
});

const triggerCollapseAll = () => collapseTrigger.value++;
</script>

<template>
  <div class="model-inspector-container">
    <div class="tabs">
      <button
        :class="['tab-btn', { active: currentTab === 'tree' }]"
        @click="currentTab = 'tree'"
      >
        Model Tree
      </button>
      <button
        :class="['tab-btn', { active: currentTab === 'links' }]"
        @click="currentTab = 'links'"
      >
        Parameter Links
      </button>
    </div>

    <div class="toolbar">
      <template v-if="currentTab === 'tree'">
        <input type="search" v-model="searchQuery" class="search-input" placeholder="Filter by path or value..." />

        <button class="icon-btn" @click="triggerCollapseAll" title="Collapse All">
          <svg viewBox="0 0 16 16" width="16" height="16" fill="currentColor">
            <path d="M2 4h12v1H2V4zm0 4h12v1H2V8zm0 4h12v1H2v-1z" />
            <path d="M4 6l4-3 4 3H4zm0 4l4 3 4-3H4z" opacity="0.6"/>
          </svg>
        </button>
      </template>

      <div class="draft-controls" v-if="isDirty">
        <button class="btn btn-discard" @click="discardDraft">Discard</button>
        <button class="btn btn-save" @click="saveDraft">Save Changes</button>
      </div>
    </div>

    <div class="view-container">

      <div class="tree-container" v-if="currentTab === 'tree'">
        <ModelNode
          v-if="draftModel.object"
          name="FitProblem"
          :value="draftModel.object"
          :parentObj="{ FitProblem: draftModel.object }"
          :references="draftModel.references"
          :start-expanded="true"
          :visible-paths="searchState.visible"
          :expand-paths="searchState.expand"
          :collapse-trigger="collapseTrigger"
        />
        <div v-else class="loading">Loading model...</div>
      </div>

      <div class="linker-container" v-else-if="currentTab === 'links'">
        <ParameterLinker
          v-if="draftModel.object"
          :draftModel="draftModel"
        />
        <div v-else class="loading">Loading model...</div>
      </div>

    </div>
  </div>
</template>

<style scoped>
.model-inspector-container {
  padding: 1rem;
  background-color: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
  display: flex;
  flex-direction: column;
  gap: 1rem;
}
.toolbar {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  border-bottom: 1px solid #eee;
  padding-bottom: 0.75rem;
}
.search-input {
  flex: 1;
  padding: 6px 8px;
  font-size: 13px;
  border: 1px solid #ccc;
  border-radius: 4px;
}
.icon-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  background: transparent;
  border: 1px solid transparent;
  border-radius: 4px;
  cursor: pointer;
}
.icon-btn:hover { background-color: #f0f0f0; border-color: #ddd; }
.draft-controls { display: flex; gap: 0.5rem; margin-left: auto; }
.btn {
  padding: 4px 10px;
  font-size: 12px;
  border-radius: 4px;
  cursor: pointer;
  border: none;
  font-weight: 600;
}
.btn-save { background-color: #007acc; color: white; }
.btn-save:hover { background-color: #005f9e; }
.btn-discard { background-color: #e0e0e0; color: #333; }
.btn-discard:hover { background-color: #ccc; }
.tree-container { overflow-y: auto; max-height: 80vh; }
.loading { color: #666; font-style: italic; }

.tabs {
  display: flex;
  gap: 2px;
  background-color: #f5f5f5;
  padding: 8px 8px 0 8px;
  border-radius: 8px 8px 0 0;
  border-bottom: 1px solid #ddd;
}
.tab-btn {
  background: transparent;
  border: 1px solid transparent;
  border-bottom: none;
  padding: 8px 16px;
  font-size: 13px;
  font-weight: 600;
  color: #666;
  cursor: pointer;
  border-radius: 6px 6px 0 0;
}
.tab-btn:hover {
  background-color: #e9e9e9;
}
.tab-btn.active {
  background-color: #fff;
  border-color: #ddd;
  color: #007acc;
  margin-bottom: -1px; /* Overlap the bottom border */
}

/* Ensure the draft controls get pushed to the right even if the search bar is hidden */
.draft-controls {
  margin-left: auto;
}
.view-container {
  display: flex;
  flex-direction: column;
  height: 100%;
}
.linker-container {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
</style>
