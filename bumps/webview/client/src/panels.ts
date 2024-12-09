import ConvergenceView from "./components/ConvergenceView.vue";
import CorrelationView from "./components/CorrelationViewPlotly.vue";
import CustomPlot from "./components/CustomPlot.vue";
import CustomUncertaintyPlot from "./components/CustomUncertaintyPlot.vue";
import DataView from "./components/DataView.vue";
import HistoryView from "./components/HistoryView.vue";
import LogView from "./components/LogView.vue";
import ModelInspect from "./components/ModelInspect.vue";
import ModelUncertaintyView from "./components/ModelUncertaintyView.vue";
import ParameterTraceView from "./components/ParameterTraceView.vue";
import ParameterView from "./components/ParameterView.vue";
import SummaryView from "./components/SummaryView.vue";
import UncertaintyView from "./components/UncertaintyView.vue";

export type Panel = { title: string; component: any };

export const panels: Panel[] = [
  { title: "Data", component: DataView },
  { title: "Summary", component: SummaryView },
  { title: "Log", component: LogView },
  { title: "History", component: HistoryView },
  { title: "Convergence", component: ConvergenceView },
  { title: "Model", component: ModelInspect },
  { title: "Parameters", component: ParameterView },
  { title: "Correlations", component: CorrelationView },
  { title: "Trace", component: ParameterTraceView },
  { title: "Model Uncertainty", component: ModelUncertaintyView },
  { title: "Uncertainty", component: UncertaintyView },
  { title: "Custom", component: CustomPlot },
  { title: "Custom Uncertainty", component: CustomUncertaintyPlot },
];
