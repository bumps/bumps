import DataView from './components/DataView.vue';
import SummaryView from './components/SummaryView.vue';
import ModelInspect from './components/ModelInspect.vue';
import ParameterView from './components/ParameterView.vue';
import LogView from './components/LogView.vue';
import ConvergenceView from './components/ConvergenceView.vue';
import CorrelationView from './components/CorrelationViewPlotly.vue';
import ParameterTraceView from './components/ParameterTraceView.vue';
import ModelUncertaintyView from './components/ModelUncertaintyView.vue';
import UncertaintyView from './components/UncertaintyView.vue';
import History from './components/History.vue';

export const panels = [
    {title: 'Data', component: DataView},
    {title: 'Summary', component: SummaryView},
    {title: 'Log', component: LogView},
    {title: 'History', component: History},
    {title: 'Convergence', component: ConvergenceView},
    {title: 'Model', component: ModelInspect},
    {title: 'Parameters', component: ParameterView},
    {title: 'Correlations', component: CorrelationView},
    {title: 'Trace', component: ParameterTraceView},
    {title: 'Model Uncertainty', component: ModelUncertaintyView},
    {title: 'Uncertainty', component: UncertaintyView},
  ];
  