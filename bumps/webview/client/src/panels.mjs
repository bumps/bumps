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
import CustomPlot from './components/CustomPlot.vue';
import History from './components/History.vue';
import CustomUncertaintyPlot from './components/CustomUncertaintyPlot.vue';
import { shared_state } from './app_state';

export const show_uncertainty = () => shared_state.value.uncertainty_available?.value?.available ?? false;
export const show_custom_plots = () => shared_state.value.custom_plots_available?.value?.parameter_based ?? false;
export const show_custom_uncertainty_plots = () => shared_state.value.custom_plots_available?.value?.uncertainty_based ?? false;

export const panels = [
    {title: 'Data', component: DataView},
    {title: 'Summary', component: SummaryView},
    {title: 'Log', component: LogView},
    {title: 'History', component: History},
    {title: 'Convergence', component: ConvergenceView},
    {title: 'Model', component: ModelInspect},
    {title: 'Parameters', component: ParameterView},
    {title: 'Correlations', component: CorrelationView, show: show_uncertainty},
    {title: 'Trace', component: ParameterTraceView, show: show_uncertainty},
    {title: 'Model Uncertainty', component: ModelUncertaintyView, show: show_uncertainty},
    {title: 'Uncertainty', component: UncertaintyView, show: show_uncertainty},
    {title: 'Custom', component: CustomPlot, show: show_custom_plots},
    {title: 'Custom Uncertainty', component: CustomUncertaintyPlot, show: show_custom_uncertainty_plots},
];