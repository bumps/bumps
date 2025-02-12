/**
 * @type {import("prettier").Config}
 * @see https://prettier.io/docs/en/configuration.html
 */

const config = {
  plugins: [
    "./node_modules/@ianvs/prettier-plugin-sort-imports/lib/src/index.js",
    "./node_modules/prettier-plugin-jsdoc/dist/index.js",
    "./node_modules/prettier-plugin-css-order/src/main.mjs",
  ],
  importOrder: ["^vue", "^[a-zA-Z]", "^@[a-zA-Z]", "^@/", "^./", "^../"],
  cssDeclarationSorterOrder: "smacss",
  jsdocCapitalizeDescription: false,
  overrides: [
    {
      files: "*.svg",
      options: {
        parser: "html",
      },
    },
  ],
  experimentalTernaries: true,
  printWidth: 120,
  // tabWidth: 4,
  // useTabs: false,
  // semi: true,
  // singleQuote: false,
  // jsxSingleQuote: false,
  // quoteProps: "as-needed",
  trailingComma: "es5",
  // bracketSpacing: true,
  // bracketSameLine: false,
  // arrowParens: "always",
  // rangeStart: 0,
  // rangeEnd: Infinity,
  // parser: "babel",
  // requirePragma: false,
  // insertPragma: false,
  // proseWrap: "preserve",
  // htmlWhitespaceSensitivity: "css",
  // vueIndentScriptAndStyle: false,
  // endOfLine: "lf",
  // embeddedLanguageFormatting: "auto",
  // singleAttributePerLine: true,
};

export default config;
