**********
BumpsPeakFit: 1d multi-peak fitting example demonstrating how to create a plugin for bumps
**********

This is an example of creating a multi-part 1D peak fitting module that has its own
errors.py calcualtions to create credible intervals for the uncertainty plotting.

If the pandas module is avialable, there is also a simple (albeit fragile) model builder module that is driven
by a .csv file of the right formatting.

In the peaks_1d.py module, there is provisions being made for a future development in bumps
such that the forward mc calculations are included in the fitness classes (e.g. peaks_1d.Peaks)
