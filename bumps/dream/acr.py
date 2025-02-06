"""
ACR upper percentiles critical value for test of single multivariate
normal outlier.

From the method given by Wilks (1963) and approaching to a F distribution
function by the Yang and Lee (1987) formulation, we compute the critical
value of the maximum squared Mahalanobis distance to detect outliers from
a normal multivariate sample.

We can generate all the critical values of the maximum squared Mahalanobis
distance presented on the Table XXXII of by Barnett and Lewis (1978) and
Table A.6 of Rencher (2002). Also with any given significance level (alpha).

Example::

    >>> print("%.4f"%ACR(3, 25, 0.01))
    13.1753

Created by::

    A. Trujillo-Ortiz, R. Hernandez-Walls, A. Castro-Perez and K. Barba-Rojo
    Facultad de Ciencias Marinas
    Universidad Autonoma de Baja California
    Apdo. Postal 453
    Ensenada, Baja California
    Mexico.
    atrujo@uabc.mx

Copyright. August 20, 2006.

To cite this file, this would be an appropriate format::

    Trujillo-Ortiz, A., R. Hernandez-Walls, A. Castro-Perez and K. Barba-Rojo.
    (2006). *ACR:Upper percentiles critical value for test of single
    multivariate  normal outlier.* A MATLAB file. [WWW document].  URL
    http://www.mathworks.com/matlabcentral/fileexchange/loadFile.do?objectId=12161

The function's name is given in honour of Dr. Alvin C. Rencher for his
invaluable contribution to multivariate statistics with his text 'Methods of
Multivariate Analysis'.

References:

[1] Barnett, V. and Lewis, T. (1978), Outliers on Statistical Data.
     New-York:John Wiley & Sons.
[2] Rencher, A. C. (2002), Methods of Multivariate Analysis. 2nd. ed.
     New-Jersey:John Wiley & Sons. Chapter 13 (pp. 408-450).
[3] Wilks, S. S. (1963), Multivariate Statistical Outliers. Sankhya,
     Series A, 25: 407-426.
[4] Yang, S. S. and Lee, Y. (1987), Identification of a Multivariate
     Outlier. Presented at the Annual  Meeting of the American
     Statistical Association, San Francisco, August 1987.
"""

__all__ = ["ACR"]

from scipy.stats import f

finv = f.ppf


def ACR(p, n, alpha=0.05):
    """
    Return critical value for test of single multivariate normal outlier
    using the Mahalanobis distance metric.

    *p* is the number of independent variables,
    *n* is the number of samples, and
    *alpha* is the significance level cutoff (default=0.05).
    """

    if alpha <= 0 or alpha >= 1:
        raise ValueError("significance level must be between 0 and 1")

    a = alpha
    # F distribution critical value with p and n-p-1 degrees of freedom
    # using the Bonferroni correction.
    fc = finv(1 - a / n, p, n - p - 1)
    result = (p * (n - 1) ** 2 * fc) / (n * (n - p - 1) + (n * p * fc))
    # = ((-1*((1/(1+(fc*p/(n-p-1))))-1))*((n-1)^2))/n;

    return result


def test():
    assert abs(ACR(3, 25, 0.01) - 13.1753251622586) < 1e-13


if __name__ == "__main__":
    test()
