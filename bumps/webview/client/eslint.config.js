import url from "url";
// import eslintPluginPrettierRecommended from "eslint-plugin-prettier/recommended";
import pluginVue from "eslint-plugin-vue";
import { FlatCompat } from "@eslint/eslintrc";
import js from "@eslint/js";
import prettierConfig from "@vue/eslint-config-prettier";
import vueTsEslintConfig from "@vue/eslint-config-typescript";

const __dirname = url.fileURLToPath(new URL(".", import.meta.url));
const compat = new FlatCompat({
  baseDirectory: __dirname,
  recommendedConfig: js.configs.recommended,
  // allConfig: js.configs.all,
});

export default [
  /** Extend recommended configs */
  ...compat.extends("plugin:vue/vue3-recommended", "plugin:vuejs-accessibility/recommended", "prettier"),
  ...pluginVue.configs["flat/recommended"],
  ...vueTsEslintConfig(),
  // eslintPluginPrettierRecommended,
  prettierConfig,
  /** Configuration */
  {
    languageOptions: {
      parserOptions: {
        ecmaVersion: "latest",
        sourceType: "script",
      },
    },
    files: ["src/**/*.js", "src/**/*.mjs", "src/**/*.ts", "src/**/*.tsx", "src/**/*.vue"],
    /** Override rules */
    rules: {
      // "max-len": ["error", { code: 120 }],
      "prefer-const": 0,
      "@typescript-eslint/ban-ts-comment": ["error", { "ts-ignore": "allow-with-description" }],
      "@typescript-eslint/no-explicit-any": "off",
      "@typescript-eslint/no-unsafe-function-type": "off",
      "@typescript-eslint/no-unused-vars": "warn",
      "prettier/prettier": [
        "warn",
        {},
        {
          usePrettierrc: true,
        },
      ],
      "vue/no-html": "off",
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
];
