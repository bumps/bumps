#!/usr/bin/env python
# This program is in the public domain.
# Authors: Paul Kienzle and James Krycka

"""
This script starts the command line interface of the Refl1D Reflectometry
Modeler application to process the command just entered.
"""

# ========================== Start program ==================================
# Process the command line that has been entered.
if __name__ == "__main__":
    # This is necessary when running the application from a frozen image and
    # using the --parallel option.  Note that freeze_support() has no effect
    # when running from a python script (i.e., in a non-frozen environment).
    import multiprocessing
    multiprocessing.freeze_support()

    import bumps.cli
    bumps.cli.main()
