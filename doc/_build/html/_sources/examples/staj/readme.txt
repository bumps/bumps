*************
MLayer Models
*************

.. contents:: :local:

This package can load models from other reflectometry fitting software.  In
this example we load an mlayer .staj file and fit the parameters within it.

The staj file can be used directly from the graphical interactor or it can
be previewed from the command line::

    $ refl1d De2_VATR.staj --preview

This shows the model plot:

.. plot::

    from sitedoc import plot_model
    plot_model('De2_VATR.staj')

and the available model parameters::

        .probe
          .back_absorption = Parameter(1, name='back_absorption')
          .background = Parameter(1e-10, name='background')
          .intensity = Parameter(1, name='intensity')
          .theta_offset = Parameter(0, name='theta_offset')
        .sample
          .layers
            [0]
              .interface = Parameter(4.24661e-11, name='B3 interface')
              .material
                .irho = Parameter(3.00904e-05, name='B3 irho')
                .rho = Parameter(5.69228, name='B3 rho')
              .thickness = Parameter(90, name='B3 thickness')
            [1]
              .interface = Parameter(4.24661e-11, name='B2 interface')
              .material
                .irho = Parameter(1.39368e-05, name='B2 irho')
                .rho = Parameter(5.86948, name='B2 rho')
              .thickness = Parameter(64.0154, name='B2 thickness')
            [2]
              .interface = Parameter(83.7958, name='B1 interface')
              .material
                .irho = Parameter(6.93684e-05, name='B1 irho')
                .rho = Parameter(0.340309, name='B1 rho')
              .thickness = Parameter(316.991, name='B1 thickness')
            [3]
              .interface = Parameter(33.2095, name='M2 interface')
              .material
                .irho = Parameter(6.93684e-05, name='M2 irho')
                .rho = Parameter(1.73106, name='M2 rho')
              .thickness = Parameter(1052.77, name='M2 thickness')
            [4]
              .interface = Parameter(20.6753, name='M1 interface')
              .material
                .irho = Parameter(0.00137419, name='M1 irho')
                .rho = Parameter(4.02059, name='M1 rho')
              .thickness = Parameter(567.547, name='M1 thickness')
            [5]
              .interface = Parameter(4.24661e-11, name='V interface')
              .material
                .irho = Parameter(0, name='V irho')
                .rho = Parameter(0, name='V rho')
              .thickness = Parameter(0, name='V thickness')
          .thickness = stack thickness:2091.32

        [chisq=2.16242, nllf=408.697]

Note that the parameters are reversed from the order in mlayer, so layer 0
is the substrate rather than the incident medium.  The graphical interactor,
refl1d_gui, allows you to adjust parameters and fit ranges before starting
the fit, but you can also do so from a script, as shown in
:download:`De2_VATR.py <De2_VATR.py>`:

.. literalinclude:: De2_VATR.py

Staj file constraints are ignored, but you can get similar functionality by
setting parameters to equal expressions of other parameters.  You can even
constrain one staj file to share parameters with another by setting, for
example::

    M1 = load_mlayer("De1_VATR.staj")
    M2 = load_mlayer("De2_VATR.staj")
    M1.sample[3].thickness = M2.sample[3].thickness
    problem = MultiFitProblem([M1,M2])
