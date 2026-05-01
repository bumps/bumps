<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { getDiff } from "json-difference";
import ModelNode from "./ModelNode.vue";
import type { AsyncSocket } from "../asyncSocket.ts";
import { setupDrawLoop } from "../setupDrawLoop";

type json = string | number | boolean | null | json[] | { [key: string]: json };

const modelJson = ref<any>({});
const searchQuery = ref("");
const collapseTrigger = ref(0);

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
  } else {
    const old_model: json = modelJson.value as Record<string, any>;
    const new_model: json = json_value as Record<string, any>;
    const diff = getDiff(old_model, new_model);
    for (let [path, oldval, newval] of diff.edited) {
      const { target, parent, key } = resolve_diffpath(old_model, path);
      if (typeof key === "number" && Array.isArray(parent)) {
        parent.splice(key, 1, newval);
      } else {
        parent[key] = newval;
      }
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

// Computes both visible and expanded paths based on the search query
const searchState = computed(() => {
  if (!searchQuery.value.trim() || !modelJson.value.object) {
    return { visible: null, expand: null };
  }

  const visiblePaths = new Set<string>();
  const expandPaths = new Set<string>();
  const query = searchQuery.value.trim().toLowerCase();
  const refs = modelJson.value.references || {};

  function traverse(obj: any, currentPath: string, ancestorMatched: boolean, key: string | number): boolean {
    let actualObj = obj;
    if (obj && typeof obj === "object" && obj.__class__ === "Reference" && refs[obj.id]) {
      actualObj = refs[obj.id];
    }

    let isMatch = false;

    // If an ancestor already matched, we ONLY check the local key/value.
    // This stops the full path string from cascading a match to all descendants.
    if (ancestorMatched) {
      isMatch =
        String(key).toLowerCase().includes(query) ||
        (typeof actualObj !== "object" && actualObj !== null && String(actualObj).toLowerCase().includes(query));
    } else {
      // If no ancestor has matched yet, we evaluate the full path string (to support "probe.mm")
      isMatch =
        currentPath.toLowerCase().includes(query) ||
        (typeof actualObj !== "object" && actualObj !== null && String(actualObj).toLowerCase().includes(query));
    }

    let descendantMatched = false;
    const childAncestorMatched = ancestorMatched || isMatch;

    if (actualObj && typeof actualObj === "object") {
      for (const childKey in actualObj) {
        if (childKey === "__class__" || childKey === "id") continue;
        const childPath = currentPath ? currentPath + "." + childKey : String(childKey);

        const childHasMatch = traverse(actualObj[childKey], childPath, childAncestorMatched, childKey);
        if (childHasMatch) descendantMatched = true;
      }
    }

    // Node is visible if it matches, a descendant matches, or an ancestor matched
    if (isMatch || descendantMatched || ancestorMatched) {
      visiblePaths.add(currentPath);
    }

    // Node is ONLY expanded if it matches or a descendant matches (prevents opening children of matches)
    if (isMatch || descendantMatched) {
      expandPaths.add(currentPath);
    }

    return isMatch || descendantMatched;
  }

  traverse(modelJson.value.object, "FitProblem", false, "FitProblem");
  return { visible: visiblePaths, expand: expandPaths };
});

const triggerCollapseAll = () => {
  collapseTrigger.value++;
};
</script>

<template>
  <div class="model-inspector-container">
    <div class="toolbar">
      <input
        v-model="searchQuery"
        type="search"
        class="search-input"
        aria-label="Filter models"
        placeholder="Filter by path or value (e.g., probe.mm or magnetism)"
      />
      <button class="action-btn" title="Collapse All" @click="triggerCollapseAll">
        <i class="bi bi-text-paragraph"></i>
      </button>
    </div>

    <div class="tree-container">
      <ModelNode
        v-if="modelJson.object"
        name="FitProblem"
        :value="modelJson.object"
        :references="modelJson.references"
        :start-expanded="true"
        :visible-paths="searchState.visible"
        :expand-paths="searchState.expand"
        :collapse-trigger="collapseTrigger"
      />
      <div v-else class="loading">Loading model...</div>
    </div>
  </div>
</template>

<style scoped>
.model-inspector-container {
  display: flex;
  flex-direction: column;
  padding: 1rem;
  gap: 1rem;
  border-radius: 8px;
  background-color: #fff;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}
.toolbar {
  display: flex;
  align-items: center;
  padding-bottom: 0.75rem;
  gap: 0.5rem;
  border-bottom: 1px solid #eee;
}
.search-input {
  flex: 1;
  padding: 4px 8px;
  border: 1px solid #ccc;
  border-radius: 4px;
  outline: none;
  font-size: 13px;
}
.search-input:focus {
  border-color: #007acc;
}
.action-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 26px;
  height: 26px;
  border: 1px solid transparent;
  border-radius: 4px;
  background: transparent;
  color: #444;
  cursor: pointer;
}
.action-btn:hover {
  border-color: #ddd;
  background-color: #f0f0f0;
}
.tree-container {
  max-height: 80vh;
  overflow-y: auto;
}
.loading {
  color: #666;
  font-family: sans-serif;
}
</style>
