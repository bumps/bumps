class Fitness:
    """
    A fitness class combines physical model, data and theory, returning
    residuals (for use in least squares fitting) and a bayesian likelihood
    (for maximum likelihood optimization).

    The fitness communicates with the optimizer via a parameter set, which
    includes initial values, ranges, aand constraints.

    A fitness function will have a set of views associated with it, including
    various plottables (axes and lines), printables (tables and text), and
    forms (panels that the GUI can use to interact with the model).
    """

class Theory:
    pass
