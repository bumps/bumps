import pluginVue from "eslint-plugin-vue";
import pluginVueA11y from "eslint-plugin-vuejs-accessibility";
import { defineConfigWithVueTs, vueTsConfigs } from "@vue/eslint-config-typescript";

export default defineConfigWithVueTs(
  /** Extend recommended configs */
  ...pluginVueA11y.configs["flat/recommended"],
  ...pluginVue.configs["flat/recommended"],
  vueTsConfigs.recommended,
  /** Configuration */
  {
    languageOptions: {
      parserOptions: {
        ecmaVersion: "latest",
        sourceType: "script",
      },
    },
    files: ["**/src/**/*.js", "**/src/**/*.mjs", "**/src/**/*.ts", "**/src/**/*.tsx", "**/src/**/*.vue"],
    ignores: ["**/node_modules/", "**/.git/", "**/dist/"],
    /** Override rules */
    rules: {
      // "max-len": ["error", { code: 120 }],
      "prefer-const": 0,
      "@typescript-eslint/ban-ts-comment": ["error", { "ts-ignore": "allow-with-description" }],
      "@typescript-eslint/no-explicit-any": "off",
      "@typescript-eslint/no-unsafe-function-type": "off",
      "@typescript-eslint/no-unused-vars": "warn",

      // --- THE "STOP FIGHTING BIOME" SECTION ---
      // Disable stylistic Vue rules that Biome handles
      "vue/max-attributes-per-line": "off",
      "vue/html-indent": "off",
      "vue/html-closing-bracket-newline": "off",
      "vue/html-self-closing": "off",
      "vue/first-attribute-linebreak": "off",
      "vue/html-closing-bracket-spacing": "off",
      "vue/singleline-html-element-content-newline": "off",
      "vue/multiline-html-element-content-newline": "off",
      // ------------------------------------------

      "vue/no-v-html": "warn",
      "vue/no-unused-vars": "warn",
      "vuejs-accessibility/label-has-for": [
        "error",
        {
          required: {
            some: ["nesting", "id"],
          },
        },
      ],
    },
  },
);
