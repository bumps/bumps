export const FITTERS = {
  dream: {
    name: "DREAM",
    settings: {
      samples: 10000,
      burn: 100,
      pop: 10,
      init: "eps",
      thin: 1,
      alpha: 0.01,
      outliers: "none",
      trim: false,
      steps: 0,
    },
  },
  lm: {
    name: "Levenberg-Marquardt",
    settings: {
      steps: 200,
      ftol: 1.5e-8,
      xtol: 1.5e-8,
    },
  },
  amoeba: {
    name: "Nelder-Mead Simplex",
    settings: {
      steps: 1000,
      starts: 1,
      radius: 0.15,
      xtol: 1e-6,
      ftol: 1e-8,
    },
  },
  de: {
    name: "Differential Evolution",
    settings: {
      steps: 1000,
      pop: 10,
      CR: 0.9,
      F: 2.0,
      ftol: 1e-8,
      xtol: 1e-6,
    },
  },
  mp: {
    name: "MPFit",
    settings: {
      steps: 200,
      ftol: 1e-10,
      xtol: 1e-10,
    },
  },
  newton: {
    name: "Quasi-Newton BFGS",
    settings: {
      steps: 3000,
      starts: 1,
      ftol: 1e-6,
      xtol: 1e-12,
    },
  },
};
