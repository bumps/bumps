.. _data-guide:

*******************
Data Representation
*******************

.. contents:: :local:

Data is x,y,dy.  Anything more complicated you will need to do yourself.

We do provide a convolution function for data in which each point has 
an independent gaussian resolution width, and a rebinning function for
1-D and 2D whcih can addjust a matrix of counts to lie on a different grid.
