<script setup lang="ts">
import { computed, ref } from "vue";

const props = defineProps<{
  draftModel: any;
}>();

const searchQuery = ref("");
const linkingParamId = ref<string | null>(null);
const linkSearchQuery = ref("");

// --- Bulk Selection State ---
const selectedIds = ref(new Set<string>());

// Helper: Generate a fallback UUID
const generateUUID = () => {
  if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    return (c === "x" ? r : (r & 0x3) | 0x8).toString(16);
  });
};

// 1. Walk the object tree and map Reference IDs to arrays of paths
const idToPaths = computed(() => {
  const map: Record<string, (string | number)[][]> = {};
  if (!props.draftModel?.object) return map;

  function walk(obj: any, currentPath: (string | number)[]) {
    if (!obj || typeof obj !== "object") return;
    if (obj.__class__ === "Reference" && obj.id) {
      if (!map[obj.id]) map[obj.id] = [];
      map[obj.id].push(currentPath);
      return;
    }
    if (Array.isArray(obj)) {
      obj.forEach((item, idx) => walk(item, [...currentPath, idx]));
    } else {
      for (const key in obj) {
        if (key === "__class__" || key === "id") continue;
        walk(obj[key], [...currentPath, key]);
      }
    }
  }

  walk(props.draftModel.object, []);
  return map;
});

const formatPath = (path: (string | number)[]) => {
  return path.reduce((acc: string, curr: string | number) => {
    if (typeof curr === "number") return `${acc}[${curr}]`;
    return acc ? `${acc}.${curr}` : String(curr);
  }, "");
};

const setValueAt = (obj: any, path: (string | number)[], value: any) => {
  let target = obj;
  for (let i = 0; i < path.length - 1; i++) target = target[path[i]];
  target[path[path.length - 1]] = value;
};

// 2. Build the flat array of all Parameters
const allParameters = computed(() => {
  if (!props.draftModel?.references) return [];
  const refs = props.draftModel.references;

  return Object.values(refs)
    .filter((r: any) => r && r.__class__ === "bumps.parameter.Parameter")
    .map((param: any) => {
      const pathsRaw = idToPaths.value[param.id] || [];
      const pathsFormatted = pathsRaw.map(formatPath);

      return {
        id: param.id,
        name: param.name || "Unnamed",
        primaryPath: pathsFormatted[0] || "Unused Parameter",
        allPathsRaw: pathsRaw,
        allPathsFormatted: pathsFormatted,
        isShared: pathsRaw.length > 1,
        rawParam: param,
      };
    });
});

// 3. Filter for the main view
const filteredParameters = computed(() => {
  let list = allParameters.value;
  if (searchQuery.value) {
    const q = searchQuery.value.toLowerCase();
    list = list.filter(
      (p) => p.name.toLowerCase().includes(q) || p.allPathsFormatted.some((path) => path.toLowerCase().includes(q)),
    );
  }
  return list.sort((a, b) => a.primaryPath.localeCompare(b.primaryPath));
});

// --- Selection Logic ---
const isAllSelected = computed(() => {
  return filteredParameters.value.length > 0 && filteredParameters.value.every((p) => selectedIds.value.has(p.id));
});

const toggleSelection = (id: string) => {
  if (selectedIds.value.has(id)) selectedIds.value.delete(id);
  else selectedIds.value.add(id);
};

const toggleAll = (e: Event) => {
  if ((e.target as HTMLInputElement).checked) {
    filteredParameters.value.forEach((p) => selectedIds.value.add(p.id));
  } else {
    filteredParameters.value.forEach((p) => selectedIds.value.delete(p.id));
  }
};

// --- Linking Logic ---
const getScoredTargets = (sourceParamId: string | null, searchString: string) => {
  let candidates = allParameters.value;
  let sourceKeywords = new Set<string>();

  if (sourceParamId) {
    const sourceParam = allParameters.value.find((p) => p.id === sourceParamId);
    if (sourceParam) {
      sourceKeywords = new Set([
        ...sourceParam.name.toLowerCase().split(/[\s_\.]+/),
        ...sourceParam.primaryPath.toLowerCase().split(/[\s_\.\[\]]+/),
      ]);
      candidates = candidates.filter((p) => p.id !== sourceParam.id);
    }
  }

  const scored = candidates.map((p) => {
    const pKeywords = [...p.name.toLowerCase().split(/[\s_\.]+/), ...p.primaryPath.toLowerCase().split(/[\s_\.\[\]]+/)];
    let score = 0;
    for (const kw of pKeywords) {
      if (kw.length > 2 && sourceKeywords.has(kw)) score++;
    }
    return { ...p, _score: score };
  });

  if (searchString) {
    const q = searchString.toLowerCase();
    return scored
      .filter((c) => c.primaryPath.toLowerCase().includes(q) || c.name.toLowerCase().includes(q))
      .sort((a, b) => b._score - a._score);
  }
  return scored.sort((a, b) => b._score - a._score);
};

const availableTargets = computed(() => getScoredTargets(linkingParamId.value, linkSearchQuery.value));

const startLinking = (paramId: string) => {
  linkingParamId.value = paramId;
  linkSearchQuery.value = "";
};

const closePopovers = () => {
  linkingParamId.value = null;
};

// Merge a specific source into a specific target
const mergeIntoTarget = (sourceId: string, targetId: string) => {
  if (sourceId === targetId) return;
  const paths = idToPaths.value[sourceId];
  if (paths) {
    paths.forEach((path) => {
      setValueAt(props.draftModel.object, path, { __class__: "Reference", id: targetId });
    });
  }
  delete props.draftModel.references[sourceId];
};

const executeLink = (targetId: string) => {
  if (linkingParamId.value) mergeIntoTarget(linkingParamId.value, targetId);
  closePopovers();
};

const executeBulkMerge = () => {
  const idsToMerge = Array.from(selectedIds.value);
  if (idsToMerge.length < 2) return;

  const survivorId = idsToMerge[0];
  for (let i = 1; i < idsToMerge.length; i++) {
    mergeIntoTarget(idsToMerge[i], survivorId);
  }

  selectedIds.value.clear();
  closePopovers();
};

const unlinkSpecificPath = (sourceId: string, rawPath: (string | number)[]) => {
  const newId = generateUUID();
  const sourceParam = props.draftModel.references[sourceId];
  const clonedParam = JSON.parse(JSON.stringify(sourceParam));

  clonedParam.id = newId;
  props.draftModel.references[newId] = clonedParam;

  setValueAt(props.draftModel.object, rawPath, { __class__: "Reference", id: newId });
};

const shatterParameter = (sourceId: string) => {
  const paths = idToPaths.value[sourceId];
  if (!paths || paths.length <= 1) return;
  for (let i = 1; i < paths.length; i++) unlinkSpecificPath(sourceId, paths[i]);
};

const executeBulkShatter = () => {
  selectedIds.value.forEach((id) => shatterParameter(id));
  selectedIds.value.clear();
};
</script>

<template>
  <div class="parameter-linker">

    <div class="toolbar">
      <input
        type="search"
        v-model="searchQuery"
        placeholder="Filter parameters by path or name..."
        class="main-search"
      />

      <div v-if="selectedIds.size > 0" class="bulk-actions">
        <div class="divider"></div>
        <span class="selection-count">{{ selectedIds.size }} selected</span>

        <button
          class="btn btn-link-action"
          :disabled="selectedIds.size < 2"
          :title="selectedIds.size < 2 ? 'Select at least 2 paths to merge' : 'Merge all selected paths to point to a single parameter'"
          @click="executeBulkMerge"
        >
          🔗 Merge Selected
        </button>

        <button
          class="btn btn-unlink-action"
          title="Unlink selected paths from their shared groups"
          @click="executeBulkShatter"
        >
          ✕ Unlink Selected
        </button>
      </div>
    </div>

    <div class="table-container">
      <table class="param-table">
        <thead>
          <tr>
            <th class="checkbox-col">
              <input type="checkbox" :checked="isAllSelected" @change="toggleAll" title="Select all filtered" />
            </th>
            <th>Parameter Path, Name, & Value</th>
            <th class="action-col">Actions</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="param in filteredParameters" :key="param.id" :class="{'is-shared': param.isShared, 'is-selected': selectedIds.has(param.id)}">

            <td class="checkbox-col">
              <input type="checkbox" :checked="selectedIds.has(param.id)" @change="toggleSelection(param.id)" />
            </td>

            <td class="path-cell" @click.self="toggleSelection(param.id)">
              <div class="name-wrapper">
                <input
                  type="text"
                  class="inline-name-input"
                  v-model="param.rawParam.name"
                  placeholder="Unnamed"
                />

                <input
                  v-if="param.rawParam.slot && param.rawParam.slot.__class__ === 'bumps.parameter.Variable'"
                  type="number"
                  class="inline-value-input"
                  v-model.number="param.rawParam.slot.value"
                />
                <span v-else class="inline-fallback">{{ param.rawParam.slot?.value ?? 'calc/expr' }}</span>

                <span v-if="param.isShared" class="shared-badge">Shared ({{ param.allPathsFormatted.length }})</span>
              </div>

              <div class="paths-list">
                <div v-for="(path, idx) in param.allPathsFormatted" :key="idx" class="path-item">
                  <span class="path-text">{{ path }}</span>
                  <button
                    v-if="param.isShared"
                    class="btn-unlink-tiny"
                    @click="unlinkSpecificPath(param.id, param.allPathsRaw[idx])"
                    title="Separate this specific path"
                  >✕</button>
                </div>
              </div>
            </td>

            <td class="status-cell" :style="{ zIndex: linkingParamId === param.id ? 50 : 'auto' }">
              <div class="link-actions">
                <button class="btn-link" @click="startLinking(param.id)">🔗 Merge into...</button>
              </div>

              <template v-if="linkingParamId === param.id">
                <div class="link-popover-overlay" @click="closePopovers"></div>
                <div class="link-popover" @click.stop>
                  <div class="popover-header">
                    <input type="search" v-model="linkSearchQuery" placeholder="Search target..." autofocus class="popover-search"/>
                  </div>
                  <ul class="suggestion-list">
                    <li v-for="target in availableTargets" :key="target.id" @click="executeLink(target.id)" class="suggestion-item">
                      <div class="s-info">
                        <span class="s-name">{{ target.name }}</span>
                        <span class="s-path">{{ target.primaryPath }}</span>
                      </div>
                      <span v-if="target._score > 0" class="s-score">★ Match</span>
                    </li>
                    <li v-if="availableTargets.length === 0" class="empty-list">No targets found.</li>
                  </ul>
                </div>
              </template>
            </td>

          </tr>
          <tr v-if="filteredParameters.length === 0">
            <td colspan="3" class="empty-state">No parameters match your search.</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<style scoped>
.parameter-linker { background: #fff; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); display: flex; flex-direction: column; height: 100%; }
.toolbar { padding: 1rem; border-bottom: 1px solid #eee; display: flex; align-items: center; gap: 12px; }
.main-search { flex: 1; padding: 8px 12px; border: 1px solid #ccc; border-radius: 6px; font-size: 14px; }
.bulk-actions { display: flex; align-items: center; gap: 8px; }
.divider { width: 1px; height: 24px; background-color: #ddd; margin: 0 4px; }
.selection-count { font-size: 13px; font-weight: bold; color: #007acc; background: #e6f3ff; padding: 4px 8px; border-radius: 4px;}
.bulk-popover-container { position: relative; }
.btn-link-action { background: #fff; border: 1px solid #ccc; padding: 6px 12px; border-radius: 4px; font-size: 13px; cursor: pointer; color: #333; font-weight: 500;}
.btn-link-action:hover { border-color: #007acc; color: #007acc; background: #f0f8ff; }
.btn-unlink-action { background: #fff; border: 1px solid #f5c2c7; padding: 6px 12px; border-radius: 4px; font-size: 13px; cursor: pointer; color: #c82a2a; font-weight: 500;}
.btn-unlink-action:hover { background: #f8d7da; }

.table-container { overflow-y: auto; flex: 1; }
.param-table { width: 100%; border-collapse: collapse; text-align: left; }
.param-table th { background: #f9f9f9; padding: 8px 12px; font-size: 12px; color: #666; border-bottom: 2px solid #eee; position: sticky; top: 0; z-index: 1; }
.param-table td { padding: 10px 12px; border-bottom: 1px solid #f0f0f0; vertical-align: top; }

.is-shared { background-color: #fafbfc; }
.is-selected { background-color: #f0f8ff !important; }
.checkbox-col { width: 40px; text-align: center; vertical-align: middle !important; }
.path-cell { max-width: 450px; cursor: pointer; }

/* Inline Editor Styling */
.name-wrapper { display: flex; align-items: center; gap: 4px; margin-bottom: 6px; }
.inline-name-input {
  font-size: 14px;
  font-weight: 600;
  color: #2c3e50;
  border: 1px solid transparent;
  background: transparent;
  border-radius: 4px;
  padding: 2px 4px;
  max-width: 160px;
  text-overflow: ellipsis;
}
.inline-name-input:hover, .inline-name-input:focus {
  background: white;
  border-color: #ccc;
  outline: none;
}
.inline-value-input { width: 80px; padding: 2px 6px; font-family: monospace; font-size: 13px; font-weight: bold; color: #881391; border: 1px solid transparent; background: transparent; border-radius: 4px; cursor: text; margin-left: 6px; }
.inline-value-input:hover, .inline-value-input:focus { background: white; border-color: #ccc; outline: none; }
.inline-fallback { font-family: monospace; font-size: 12px; color: #888; background: #f0f0f0; padding: 2px 6px; border-radius: 4px; pointer-events: none; margin-left: 6px; }
.shared-badge { font-size: 10px; background-color: #e6f3ff; color: #005f9e; padding: 2px 6px; border-radius: 4px; font-weight: bold; text-transform: uppercase; pointer-events: none; margin-left: 6px; }

.paths-list { display: flex; flex-direction: column; gap: 2px; pointer-events: none; }
.path-item { display: flex; align-items: center; gap: 6px; pointer-events: auto; }
.path-text { font-family: monospace; font-size: 11px; color: #666; word-break: break-all; }
.btn-unlink-tiny { background: transparent; border: none; color: #c82a2a; cursor: pointer; padding: 0 4px; font-size: 10px; border-radius: 3px; }
.btn-unlink-tiny:hover { background: #f8d7da; }

.action-col { width: 120px; }
.status-cell { position: relative; vertical-align: middle !important; }
.btn-link { background: white; border: 1px dashed #ccc; padding: 6px 12px; font-size: 12px; border-radius: 4px; color: #666; cursor: pointer; width: 100%; text-align: left; }
.btn-link:hover { border-color: #007acc; color: #007acc; background: #f0f8ff; }

.empty-state { text-align: center; padding: 2rem; color: #888; font-style: italic; }

/* Popover UI */
.link-popover-overlay { position: fixed; top: 0; left: 0; right: 0; bottom: 0; z-index: 99; }
.link-popover { position: absolute; top: 100%; right: 0; width: 350px; background: white; border: 1px solid #ccc; border-radius: 6px; box-shadow: 0 4px 16px rgba(0,0,0,0.15); z-index: 100; margin-top: 4px; }
.bulk-popover { left: 0; width: 400px; }
.popover-search { width: 100%; box-sizing: border-box; padding: 8px; border: none; border-bottom: 1px solid #eee; border-radius: 6px 6px 0 0; outline: none; }
.suggestion-list { list-style: none; margin: 0; padding: 0; max-height: 250px; overflow-y: auto; }
.suggestion-item { padding: 8px; border-bottom: 1px solid #f5f5f5; cursor: pointer; display: flex; justify-content: space-between; align-items: center; }
.suggestion-item:hover { background: #f0f8ff; }
.s-info { display: flex; flex-direction: column; gap: 2px; }
.s-name { font-size: 13px; font-weight: 500; }
.s-path { font-size: 10px; font-family: monospace; color: #888; }
.s-score { font-size: 10px; color: #d9822b; font-weight: bold; background: #fff5e6; padding: 2px 4px; border-radius: 4px; }
.empty-list { padding: 12px; text-align: center; color: #888; font-style: italic; }
</style>
