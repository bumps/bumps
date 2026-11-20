/** global options for vue-tippy */
export const options = {
  /** register as v-tooltip / <tooltip> (vue-tippy defaults to v-tippy / <tippy>) */
  directive: "tooltip",
  component: "tooltip",
  defaultProps: {
    delay: [300, 0],
    duration: 200,
  },
};
