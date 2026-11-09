<!-- Script editor tab -->

<template>
  <div id="editor" class="container" style="height: 100%; width: 100%;">
    <div class="d-flex justify-content-between mb-2">
      <div>
        <button type="button" class="btn btn-primary btn-sm me-2" @click="download">Download</button>
        <button type="button" class="btn btn-secondary btn-sm me-2" @click="uploadFile">Upload</button>
        <button type="button" class="btn btn-danger btn-sm me-2" @click="content = DEFAULT_CONTENT">Reset</button>
      </div>
      <div>
        <button
          type="button"
          class="btn btn-success btn-sm"
          @click="processScript"
          v-tooltip="'Send the script to the backend for processing'"
        >
          Process
        </button>
      </div>
    </div>
    <div id="code-wrapper" style="padding: 0.5rem; height: calc(100% - 2rem); width: 100%;">
      <CodeEditor v-model:value="content" :theme="monacoTheme" :options="editorOptions" />
    </div>
  </div>
</template>

<script lang="ts" setup>
import { editor as monacoEditor } from "monaco-editor";
import { CodeEditor } from "monaco-editor-vue3";
import { computed, ref, watch } from "vue";
import { theme } from "@/app_state";
import "@/global/monaco";

// CodeEditor only reads `theme` on creation, so push changes via setTheme.
const monacoTheme = computed(() => (theme.value === "dark" ? "vs-dark" : "vs"));
watch(monacoTheme, (t) => monacoEditor.setTheme(t));

const editorOptions = {
  fontSize: 14,
  minimap: { enabled: false },
  automaticLayout: true,
  language: "python",
};

const DEFAULT_CONTENT = `print("Welcome to the Bumps Editor!")`;

const content = ref(DEFAULT_CONTENT);

function download() {
  const blob = new Blob([content.value], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "script.py";
  a.click();
  URL.revokeObjectURL(url);
}

function uploadFile() {
  const input = document.createElement("input");
  input.type = "file";
  input.accept = ".py,.txt";
  input.onchange = (event: Event) => {
    const target = event.target as HTMLInputElement;
    if (target.files && target.files.length > 0) {
      const file = target.files[0];
      const reader = new FileReader();
      reader.onload = (e: ProgressEvent<FileReader>) => {
        if (e.target && typeof e.target.result === "string") {
          content.value = e.target.result;
        }
      };
      reader.readAsText(file);
    }
  };
  input.click();
}

function processScript() {
  // Placeholder for sending content to backend
  console.log("Sending content to backend:", content.value);
}
</script>

<style scoped>
.container {
  align-self: center;
  width: 90%;
  padding: 1rem;
}
</style>
