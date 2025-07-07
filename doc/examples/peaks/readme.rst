.. _peaks-example:

Peak Fitting
************

This example shows how to develop multipart models using bumps parameters.
The data format is 2D, so the usual 1D x-y plots are not sufficient, and
a special plot method is needed to display the data.

This uses a library of peak functions to model the peaks. In order for this
to work peaks.py must be on your python path. For example, on linux or mac:

    PYTHONPATH=. bumps -b --fit=dream model.py --store=peaks.h5
