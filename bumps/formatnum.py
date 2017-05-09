# This program is public domain
# Author: Paul Kienzle
"""
Format values and uncertainties nicely for printing.

The formatted value uses only the number of digits warranted by
the uncertainty in the measurement.

:func:`format_value` shows the value without the uncertainty.

:func:`format_uncertainty_pm` shows the expanded format v +/- err.

:func:`format_uncertainty_compact` shows the compact format v(##),
where the number in parenthesis is the uncertainty in the last two digits of v.

:func:`format_uncertainty` uses the compact format by default, but this
can be changed to use the expanded +/- format by setting
format_uncertainty.compact to False.  This is a global setting which should
be considered a user preference.  Any library code that depends on a specific
format style should use the corresponding formatting function.

If the uncertainty is 0 or not otherwise provided, the simple
%g floating point format option is used.

Infinite and indefinite numbers are represented as inf and NaN.

Example::

    >>> v,dv = 757.2356,0.01032
    >>> print(format_uncertainty_pm(v,dv))
    757.236 +/- 0.010
    >>> print(format_uncertainty_compact(v,dv))
    757.236(10)
    >>> print(format_uncertainty(v,dv))
    757.236(10)
    >>> format_uncertainty.compact = False
    >>> print(format_uncertainty(v,dv))
    757.236 +/- 0.010
"""
from __future__ import division
import math

from numpy import isinf, isnan, inf, NaN

__all__ = ['format_value', 'format_uncertainty',
           'format_uncertainty_compact', 'format_uncertainty_pm',
          ]


# Coordinating scales across a set of numbers is not supported.  For easy
# comparison a set of numbers should be shown in the same scale.  One could
# force this from the outside by adding scale parameter (either 10**n, n, or
# a string representing the desired SI prefix) and having a separate routine
# which computes the scale given a set of values.

# Coordinating scales with units offers its own problems.  Again, the user
# may want to force particular units.  This can be done by outside of the
# formatting routines by scaling the numbers to the appropriate units then
# forcing them to print with scale 10**0.  If this is a common operation,
# however, it may want to happen inside.

# The value e<n> is currently formatted into the number.  Alternatively this
# scale factor could be returned so that the user can choose the appropriate
# SI prefix when printing the units.  This gets tricky when talking about
# composite units such as 2.3e-3 m**2 -> 2300 mm**2, and with volumes
# such as 1 g/cm**3 -> 1 kg/L.


def format_value(value, uncertainty):
    """
    Given *value* v and *uncertainty* dv, return a string v which is
    the value formatted with the appropriate number of digits.
    """
    return _format_uncertainty(value, uncertainty, compact=None)


def format_uncertainty_pm(value, uncertainty):
    """
    Given *value* v and *uncertainty* dv, return a string v +/- dv.
    """
    return _format_uncertainty(value, uncertainty, compact=False)


def format_uncertainty_compact(value, uncertainty):
    """
    Given *value* v and *uncertainty* dv, return the compact
    representation v(##), where ## are the first two digits of
    the uncertainty.
    """
    return _format_uncertainty(value, uncertainty, compact=True)


def format_uncertainty(value, uncertainty):
    """
    Value and uncertainty formatter.

    Either the expanded v +/- dv form or the compact v(##) form will be
    used depending on whether *format_uncertainty.compact* is True or False.
    The default is True.
    """
    return _format_uncertainty(value, uncertainty, format_uncertainty.compact)
format_uncertainty.compact = True

def _format_uncertainty(value, uncertainty, compact):
    """
    Implementation of both the compact and the +/- formats.
    """
    # Handle indefinite value
    if isinf(value):
        return "inf" if value > 0 else "-inf"
    if isnan(value):
        return "NaN"

    # Handle indefinite uncertainty
    if uncertainty is None or uncertainty <= 0 or isnan(uncertainty):
        return "%g" % value
    if isinf(uncertainty):
        if compact is None:
            return "%.2g" % value
        elif compact:
            return "%.2g(inf)" % value
        else:
            return "%.2g +/- inf" % value

    # Handle zero and negative values
    sign = "-" if value < 0 else ""
    value = abs(value)

    # Determine scale of value and error
    err_place = int(math.floor(math.log10(uncertainty)))
    if value == 0:
        val_place = err_place - 1
    else:
        val_place = int(math.floor(math.log10(value)))

    if err_place > val_place:
        # Degenerate case: error bigger than value
        # The mantissa is 0.#(##)e#, 0.0#(##)e# or 0.00#(##)e#
        val_place = err_place + 2
    elif err_place == val_place:
        # Degenerate case: error and value the same order of magnitude
        # The value is ##(##)e#, #.#(##)e# or 0.##(##)e#
        val_place = err_place + 1
    elif err_place <= 1 and val_place >= -3:
        # Normal case: nice numbers and errors
        # The value is ###.###(##)
        val_place = 0
    else:
        # Extreme cases: zeros before value or after error
        # The value is ###.###(##)e#, ##.####(##)e# or #.#####(##)e#
        pass

    # Force engineering notation, with exponent a multiple of 3
    val_place = int(math.floor(val_place / 3.)) * 3

    # Format the result
    digits_after_decimal = abs(val_place - err_place + 1)
    # Only use one digit of uncertainty if no precision included in result
    #if compact is None: digits_after_decimal -= 1
    val_str = "%.*f" % (digits_after_decimal, value / 10. ** val_place)
    exp_str = "e%d" % val_place if val_place != 0 else ""
    if compact is None:
        result = "".join((sign, val_str, exp_str))
    elif compact:
        err_str = "(%2d)" % int(uncertainty / 10. ** (err_place - 1) + 0.5)
        result = "".join((sign, val_str, err_str, exp_str))
    else:
        err_str = "%.*f" % (digits_after_decimal,
                            uncertainty / 10. ** val_place)
        result = "".join((sign, val_str, exp_str + " +/- ", err_str, exp_str))
    # print sign,value, uncertainty, "=>", result
    return result


def test_compact():
    # Oops... renamed function after writing tests
    value_str = format_uncertainty_compact

    # val_place > err_place
    assert value_str(1235670, 766000) == "1.24(77)e6"
    assert value_str(123567., 76600) == "124(77)e3"
    assert value_str(12356.7, 7660) == "12.4(77)e3"
    assert value_str(1235.67, 766) == "1.24(77)e3"
    assert value_str(123.567, 76.6) == "124(77)"
    assert value_str(12.3567, 7.66) == "12.4(77)"
    assert value_str(1.23567, .766) == "1.24(77)"
    assert value_str(.123567, .0766) == "0.124(77)"
    assert value_str(.0123567, .00766) == "0.0124(77)"
    assert value_str(.00123567, .000766) == "0.00124(77)"
    assert value_str(.000123567, .0000766) == "124(77)e-6"
    assert value_str(.0000123567, .00000766) == "12.4(77)e-6"
    assert value_str(.00000123567, .000000766) == "1.24(77)e-6"
    assert value_str(.000000123567, .0000000766) == "124(77)e-9"
    assert value_str(.00000123567, .0000000766) == "1.236(77)e-6"
    assert value_str(.0000123567, .0000000766) == "12.357(77)e-6"
    assert value_str(.000123567, .0000000766) == "123.567(77)e-6"
    assert value_str(.00123567, .000000766) == "0.00123567(77)"
    assert value_str(.0123567, .00000766) == "0.0123567(77)"
    assert value_str(.123567, .0000766) == "0.123567(77)"
    assert value_str(1.23567, .000766) == "1.23567(77)"
    assert value_str(12.3567, .00766) == "12.3567(77)"
    assert value_str(123.567, .0764) == "123.567(76)"
    assert value_str(1235.67, .764) == "1235.67(76)"
    assert value_str(12356.7, 7.64) == "12356.7(76)"
    assert value_str(123567, 76.4) == "123567(76)"
    assert value_str(1235670, 764) == "1.23567(76)e6"
    assert value_str(12356700, 764) == "12.35670(76)e6"
    assert value_str(123567000, 764) == "123.56700(76)e6"
    assert value_str(123567000, 7640) == "123.5670(76)e6"
    assert value_str(1235670000, 76400) == "1.235670(76)e9"

    # val_place == err_place
    assert value_str(123567, 764000) == "0.12(76)e6"
    assert value_str(12356.7, 76400) == "12(76)e3"
    assert value_str(1235.67, 7640) == "1.2(76)e3"
    assert value_str(123.567, 764) == "0.12(76)e3"
    assert value_str(12.3567, 76.4) == "12(76)"
    assert value_str(1.23567, 7.64) == "1.2(76)"
    assert value_str(.123567, .764) == "0.12(76)"
    assert value_str(.0123567, .0764) == "12(76)e-3"
    assert value_str(.00123567, .00764) == "1.2(76)e-3"
    assert value_str(.000123567, .000764) == "0.12(76)e-3"

    # val_place == err_place-1
    assert value_str(123567, 7640000) == "0.1(76)e6"
    assert value_str(12356.7, 764000) == "0.01(76)e6"
    assert value_str(1235.67, 76400) == "0.001(76)e6"
    assert value_str(123.567, 7640) == "0.1(76)e3"
    assert value_str(12.3567, 764) == "0.01(76)e3"
    assert value_str(1.23567, 76.4) == "0.001(76)e3"
    assert value_str(.123567, 7.64) == "0.1(76)"
    assert value_str(.0123567, .764) == "0.01(76)"
    assert value_str(.00123567, .0764) == "0.001(76)"
    assert value_str(.000123567, .00764) == "0.1(76)e-3"

    # val_place == err_place-2
    assert value_str(12356700, 7640000000) == "0.0(76)e9"
    assert value_str(1235670, 764000000) == "0.00(76)e9"
    assert value_str(123567, 76400000) == "0.000(76)e9"
    assert value_str(12356, 7640000) == "0.0(76)e6"
    assert value_str(1235, 764000) == "0.00(76)e6"
    assert value_str(123, 76400) == "0.000(76)e6"
    assert value_str(12, 7640) == "0.0(76)e3"
    assert value_str(1, 764) == "0.00(76)e3"
    assert value_str(0.1, 76.4) == "0.000(76)e3"
    assert value_str(0.01, 7.64) == "0.0(76)"
    assert value_str(0.001, 0.764) == "0.00(76)"
    assert value_str(0.0001, 0.0764) == "0.000(76)"
    assert value_str(0.00001, 0.00764) == "0.0(76)e-3"

    # val_place == err_place-3
    assert value_str(12356700, 76400000000) == "0.000(76)e12"
    assert value_str(1235670, 7640000000) == "0.0(76)e9"
    assert value_str(123567, 764000000) == "0.00(76)e9"
    assert value_str(12356, 76400000) == "0.000(76)e9"
    assert value_str(1235, 7640000) == "0.0(76)e6"
    assert value_str(123, 764000) == "0.00(76)e6"
    assert value_str(12, 76400) == "0.000(76)e6"
    assert value_str(1, 7640) == "0.0(76)e3"
    assert value_str(0.1, 764) == "0.00(76)e3"
    assert value_str(0.01, 76.4) == "0.000(76)e3"
    assert value_str(0.001, 7.64) == "0.0(76)"
    assert value_str(0.0001, 0.764) == "0.00(76)"
    assert value_str(0.00001, 0.0764) == "0.000(76)"
    assert value_str(0.000001, 0.00764) == "0.0(76)e-3"

    # Zero values
    assert value_str(0, 7640000) == "0.0(76)e6"
    assert value_str(0, 764000) == "0.00(76)e6"
    assert value_str(0,  76400) == "0.000(76)e6"
    assert value_str(0,   7640) == "0.0(76)e3"
    assert value_str(0,    764) == "0.00(76)e3"
    assert value_str(0,     76.4) == "0.000(76)e3"
    assert value_str(0,      7.64) == "0.0(76)"
    assert value_str(0,      0.764) == "0.00(76)"
    assert value_str(0,      0.0764) == "0.000(76)"
    assert value_str(0,      0.00764) == "0.0(76)e-3"
    assert value_str(0,      0.000764) == "0.00(76)e-3"
    assert value_str(0,      0.0000764) == "0.000(76)e-3"

    # negative values
    assert value_str(-1235670, 765000) == "-1.24(77)e6"
    assert value_str(-1.23567, .766) == "-1.24(77)"
    assert value_str(-.00000123567, .0000000766) == "-1.236(77)e-6"
    assert value_str(-12356.7, 7.64) == "-12356.7(76)"
    assert value_str(-123.567, 764) == "-0.12(76)e3"
    assert value_str(-1235.67, 76400) == "-0.001(76)e6"
    assert value_str(-.000123567, .00764) == "-0.1(76)e-3"
    assert value_str(-12356, 7640000) == "-0.0(76)e6"
    assert value_str(-12, 76400) == "-0.000(76)e6"
    assert value_str(-0.0001, 0.764) == "-0.00(76)"

    # non-finite values
    assert value_str(-inf, None) == "-inf"
    assert value_str(inf, None) == "inf"
    assert value_str(NaN, None) == "NaN"

    # bad or missing uncertainty
    assert value_str(-1.23567, NaN) == "-1.23567"
    assert value_str(-1.23567, -inf) == "-1.23567"
    assert value_str(-1.23567, -0.1) == "-1.23567"
    assert value_str(-1.23567, 0) == "-1.23567"
    assert value_str(-1.23567, None) == "-1.23567"
    assert value_str(-1.23567, inf) == "-1.2(inf)"


def test_pm():
    # Oops... renamed function after writing tests
    value_str = format_uncertainty_pm

    # val_place > err_place
    assert value_str(1235670, 766000) == "1.24e6 +/- 0.77e6"
    assert value_str(123567., 76600) == "124e3 +/- 77e3"
    assert value_str(12356.7,  7660) == "12.4e3 +/- 7.7e3"
    assert value_str(1235.67,   766) == "1.24e3 +/- 0.77e3"
    assert value_str(123.567,    76.6) == "124 +/- 77"
    assert value_str(12.3567,     7.66) == "12.4 +/- 7.7"
    assert value_str(1.23567,      .766) == "1.24 +/- 0.77"
    assert value_str(.123567,      .0766) == "0.124 +/- 0.077"
    assert value_str(.0123567,     .00766) == "0.0124 +/- 0.0077"
    assert value_str(.00123567,    .000766) == "0.00124 +/- 0.00077"
    assert value_str(.000123567,   .0000766) == "124e-6 +/- 77e-6"
    assert value_str(.0000123567,  .00000766) == "12.4e-6 +/- 7.7e-6"
    assert value_str(.00000123567, .000000766) == "1.24e-6 +/- 0.77e-6"
    assert value_str(.000000123567, .0000000766) == "124e-9 +/- 77e-9"
    assert value_str(.00000123567, .0000000766) == "1.236e-6 +/- 0.077e-6"
    assert value_str(.0000123567,  .0000000766) == "12.357e-6 +/- 0.077e-6"
    assert value_str(.000123567,   .0000000766) == "123.567e-6 +/- 0.077e-6"
    assert value_str(.00123567,    .000000766) == "0.00123567 +/- 0.00000077"
    assert value_str(.0123567,     .00000766) == "0.0123567 +/- 0.0000077"
    assert value_str(.123567,      .0000766) == "0.123567 +/- 0.000077"
    assert value_str(1.23567,      .000766) == "1.23567 +/- 0.00077"
    assert value_str(12.3567,      .00766) == "12.3567 +/- 0.0077"
    assert value_str(123.567,      .0764) == "123.567 +/- 0.076"
    assert value_str(1235.67,      .764) == "1235.67 +/- 0.76"
    assert value_str(12356.7,     7.64) == "12356.7 +/- 7.6"
    assert value_str(123567,     76.4) == "123567 +/- 76"
    assert value_str(1235670,   764) == "1.23567e6 +/- 0.00076e6"
    assert value_str(12356700,  764) == "12.35670e6 +/- 0.00076e6"
    assert value_str(123567000, 764) == "123.56700e6 +/- 0.00076e6"
    assert value_str(123567000, 7640) == "123.5670e6 +/- 0.0076e6"
    assert value_str(1235670000, 76400) == "1.235670e9 +/- 0.000076e9"

    # val_place == err_place
    assert value_str(123567, 764000) == "0.12e6 +/- 0.76e6"
    assert value_str(12356.7, 76400) == "12e3 +/- 76e3"
    assert value_str(1235.67, 7640) == "1.2e3 +/- 7.6e3"
    assert value_str(123.567, 764) == "0.12e3 +/- 0.76e3"
    assert value_str(12.3567, 76.4) == "12 +/- 76"
    assert value_str(1.23567, 7.64) == "1.2 +/- 7.6"
    assert value_str(.123567, .764) == "0.12 +/- 0.76"
    assert value_str(.0123567, .0764) == "12e-3 +/- 76e-3"
    assert value_str(.00123567, .00764) == "1.2e-3 +/- 7.6e-3"
    assert value_str(.000123567, .000764) == "0.12e-3 +/- 0.76e-3"

    # val_place == err_place-1
    assert value_str(123567, 7640000) == "0.1e6 +/- 7.6e6"
    assert value_str(12356.7, 764000) == "0.01e6 +/- 0.76e6"
    assert value_str(1235.67, 76400) == "0.001e6 +/- 0.076e6"
    assert value_str(123.567, 7640) == "0.1e3 +/- 7.6e3"
    assert value_str(12.3567, 764) == "0.01e3 +/- 0.76e3"
    assert value_str(1.23567, 76.4) == "0.001e3 +/- 0.076e3"
    assert value_str(.123567, 7.64) == "0.1 +/- 7.6"
    assert value_str(.0123567, .764) == "0.01 +/- 0.76"
    assert value_str(.00123567, .0764) == "0.001 +/- 0.076"
    assert value_str(.000123567, .00764) == "0.1e-3 +/- 7.6e-3"

    # val_place == err_place-2
    assert value_str(12356700, 7640000000) == "0.0e9 +/- 7.6e9"
    assert value_str(1235670, 764000000) == "0.00e9 +/- 0.76e9"
    assert value_str(123567, 76400000) == "0.000e9 +/- 0.076e9"
    assert value_str(12356, 7640000) == "0.0e6 +/- 7.6e6"
    assert value_str(1235, 764000) == "0.00e6 +/- 0.76e6"
    assert value_str(123, 76400) == "0.000e6 +/- 0.076e6"
    assert value_str(12, 7640) == "0.0e3 +/- 7.6e3"
    assert value_str(1, 764) == "0.00e3 +/- 0.76e3"
    assert value_str(0.1, 76.4) == "0.000e3 +/- 0.076e3"
    assert value_str(0.01, 7.64) == "0.0 +/- 7.6"
    assert value_str(0.001, 0.764) == "0.00 +/- 0.76"
    assert value_str(0.0001, 0.0764) == "0.000 +/- 0.076"
    assert value_str(0.00001, 0.00764) == "0.0e-3 +/- 7.6e-3"

    # val_place == err_place-3
    assert value_str(12356700, 76400000000) == "0.000e12 +/- 0.076e12"
    assert value_str(1235670, 7640000000) == "0.0e9 +/- 7.6e9"
    assert value_str(123567, 764000000) == "0.00e9 +/- 0.76e9"
    assert value_str(12356, 76400000) == "0.000e9 +/- 0.076e9"
    assert value_str(1235, 7640000) == "0.0e6 +/- 7.6e6"
    assert value_str(123, 764000) == "0.00e6 +/- 0.76e6"
    assert value_str(12, 76400) == "0.000e6 +/- 0.076e6"
    assert value_str(1, 7640) == "0.0e3 +/- 7.6e3"
    assert value_str(0.1, 764) == "0.00e3 +/- 0.76e3"
    assert value_str(0.01, 76.4) == "0.000e3 +/- 0.076e3"
    assert value_str(0.001, 7.64) == "0.0 +/- 7.6"
    assert value_str(0.0001, 0.764) == "0.00 +/- 0.76"
    assert value_str(0.00001, 0.0764) == "0.000 +/- 0.076"
    assert value_str(0.000001, 0.00764) == "0.0e-3 +/- 7.6e-3"

    # Zero values
    assert value_str(0, 7640000) == "0.0e6 +/- 7.6e6"
    assert value_str(0, 764000) == "0.00e6 +/- 0.76e6"
    assert value_str(0,  76400) == "0.000e6 +/- 0.076e6"
    assert value_str(0,   7640) == "0.0e3 +/- 7.6e3"
    assert value_str(0,    764) == "0.00e3 +/- 0.76e3"
    assert value_str(0,     76.4) == "0.000e3 +/- 0.076e3"
    assert value_str(0,      7.64) == "0.0 +/- 7.6"
    assert value_str(0,      0.764) == "0.00 +/- 0.76"
    assert value_str(0,      0.0764) == "0.000 +/- 0.076"
    assert value_str(0,      0.00764) == "0.0e-3 +/- 7.6e-3"
    assert value_str(0,      0.000764) == "0.00e-3 +/- 0.76e-3"
    assert value_str(0,      0.0000764) == "0.000e-3 +/- 0.076e-3"

    # negative values
    assert value_str(-1235670, 766000) == "-1.24e6 +/- 0.77e6"
    assert value_str(-1.23567, .766) == "-1.24 +/- 0.77"
    assert value_str(-.00000123567, .0000000766) == "-1.236e-6 +/- 0.077e-6"
    assert value_str(-12356.7, 7.64) == "-12356.7 +/- 7.6"
    assert value_str(-123.567, 764) == "-0.12e3 +/- 0.76e3"
    assert value_str(-1235.67, 76400) == "-0.001e6 +/- 0.076e6"
    assert value_str(-.000123567, .00764) == "-0.1e-3 +/- 7.6e-3"
    assert value_str(-12356, 7640000) == "-0.0e6 +/- 7.6e6"
    assert value_str(-12, 76400) == "-0.000e6 +/- 0.076e6"
    assert value_str(-0.0001, 0.764) == "-0.00 +/- 0.76"

    # non-finite values
    assert value_str(-inf, None) == "-inf"
    assert value_str(inf, None) == "inf"
    assert value_str(NaN, None) == "NaN"

    # bad or missing uncertainty
    assert value_str(-1.23567, NaN) == "-1.23567"
    assert value_str(-1.23567, -inf) == "-1.23567"
    assert value_str(-1.23567, -0.1) == "-1.23567"
    assert value_str(-1.23567, 0) == "-1.23567"
    assert value_str(-1.23567, None) == "-1.23567"
    assert value_str(-1.23567, inf) == "-1.2 +/- inf"


def test():
    # Check compact and plus/minus formats
    test_compact()
    test_pm()
    # Check that the default is the compact format
    assert format_uncertainty(-1.23567, 0.766) == "-1.24(77)"

    import doctest
    doctest.testmod()

if __name__ == "__main__":
    test()
