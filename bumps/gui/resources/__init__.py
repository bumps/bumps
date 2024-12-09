# Identify the resource files for the bumps GUI
from ..resfiles import Resources

resources = Resources(
    package=__name__,
    patterns=("*.png", "*.jpg", "*.ico", "*.wav"),
    datadir="bumps-data",
    check_file="reload.png",
    env="BUMPS_DATA",
)
del Resources
