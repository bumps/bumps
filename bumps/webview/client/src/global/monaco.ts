import EditorWorker from "monaco-editor/editor/editor.worker?worker";

/**
 * Tell Monaco how to start its web worker. Without this it tries to load the
 * worker from a relative URL that Vite doesn't bundle, and logs errors.
 * Python has no dedicated language worker, so the base editor worker covers
 * everything we use.
 */
self.MonacoEnvironment = {
  getWorker: () => new EditorWorker(),
};
