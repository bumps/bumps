# persist settings to disk

import sys
from pathlib import Path
from typing import Optional

ETC_PATH = Path(sys.prefix, "etc")
DEFAULT_APPLICATION = "bumps"


def set_value(key: str, value: str, application: str = DEFAULT_APPLICATION):
    settings_path = ETC_PATH / application / key
    try:
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        with open(settings_path, "w") as f:
            f.write(value)
        return True
    except Exception as e:
        return False


def get_value(key: str, default_value: Optional[str] = None, application: str = DEFAULT_APPLICATION):
    settings_path = ETC_PATH / application / key
    try:
        with open(settings_path, "r") as f:
            return f.read()
    except Exception as e:
        return default_value
