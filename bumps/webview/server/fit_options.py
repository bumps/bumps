from typing import Dict, List, Optional
from . import api

FIT_FIELDS = {
    "starts": ["Starts", "integer"],
    "steps": ["Steps", "integer"],
    "samples": ["Samples", "integer"],
    "xtol": ["x tolerance", "float"],
    "ftol": ["f(x) tolerance", "float"],
    "alpha": ["Convergence", "float"],
    "stop": ["Stopping criteria", "string"],
    "thin": ["Thinning", "integer"],
    "burn": ["Burn-in steps", "integer"],
    "pop": ["Population", "float"],
    "init": ["Initializer", ["eps", "lhs", "cov", "random"]],
    "CR": ["Crossover ratio", "float"],
    "F": ["Scale", "float"],
    "nT": ["# Temperatures", "integer"],
    "Tmin": ["Min temperature", "float"],
    "Tmax": ["Max temperature", "float"],
    "radius": ["Simplex radius", "float"],
    "trim": ["Burn-in trim", "boolean"],
    "outliers": ["Outliers", ["none", "iqr", "grubbs", "mahal"]],
}


def parse_fit_options(fitter_id: str, fit_options: Optional[List[str]] = None) -> Dict:
    if fitter_id not in api.FITTER_DEFAULTS:
        raise ValueError(f"invalid fitter: {fitter_id}")
    fitter_settings: Dict = api.FITTER_DEFAULTS[fitter_id]["settings"]
    if fit_options is not None:
        # fit options is a list of strings of the form "key=value"
        for option_str in fit_options:
            parts = option_str.split("=")
            if len(parts) != 2:
                raise ValueError(f"invalid fit option: {option_str}, must be of form 'key=value'")
            key, value = parts
            if key not in fitter_settings:
                raise ValueError(
                    f"invalid fit option: '{key}' for fitter '{fitter_id}'; valid options are: {list(fitter_settings.keys())}"
                )
            _label, parse_type = FIT_FIELDS[key]
            if parse_type == "integer":
                value = int(value)
            elif parse_type == "float":
                value = float(value)
            elif parse_type == "string":
                pass
            elif parse_type == "boolean":
                if value.lower() in ["true", "1", "yes", "on"]:
                    value = True
                elif value.lower() in ["false", "0", "no", "off"]:
                    value = False
                else:
                    raise ValueError(f"invalid value for {key}: '{value}'; valid options are yes and no")
            elif isinstance(parse_type, list):
                if value not in parse_type:
                    raise ValueError(f"invalid value for {key}: '{value}'; valid options are: {parse_type}")
            else:
                raise ValueError(f"invalid type: {parse_type}")

            fitter_settings[key] = value

    return fitter_settings
