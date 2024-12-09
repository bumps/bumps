# This program is public domain
# Author: Paul Kienzle
"""
Conditional expression manager

Conditional expressions are formed from combinations of
and (&), or (|), exclusive or (^) and not  (!), as well
as primitive tests.   The resulting expressions can be
evaluated on particular inputs, with the inputs passed
down to the primitive tests.  Primitive tests should be
subclassed from Condition, with the __call__  method
defined to return the truth value of the condition
given the inputs.

For example, the following creates a test whether a
number is in the open interval (0,1)::

    >>> class gt(Condition):
    ...    def __init__(self, base): self.base = base
    ...    def __call__(self, test): return test>self.base
    ...    def __str__(self): return "x>%g"%self.base
    >>> class lt(Condition):
    ...    def __init__(self, base): self.base = base
    ...    def __call__(self, test): return test<self.base
    ...    def __str__(self): return "x<%g"%self.base
    >>> test = lt(1) & gt(0)
    >>> print(test)
    (x<1 and x>0)
    >>> test(0.5)
    True
    >>> test(1)
    False

No assumptions are made about the structure of the arguments
to the primitives, but all primitives in an expression should
accept the same arguments.

The constants true and false are predefined as primitives
which can take any arguments::

    >>> true()
    True
    >>> false("this",string="ignored")
    False

You can find the individual terms of the expression using
the method primitives:

    >>> (true&false).primitives() == set([true,false])
    True

In many instances you not only want to know that an expression
is true or false, but why it is true or false. The status method
on expressions does this, returning not only the truth status as
a boolean, but also a list of primitives and Not(primitives)
saying which conditions are responsible.  For example::

    >>> class iters(Condition):
    ...    def __init__(self, base): self.base = base
    ...    def __call__(self, state): return state['iters']>self.base
    ...    def __str__(self): return "iters>%d"%self.base
    >>> class stepsize(Condition):
    ...    def __init__(self, base): self.base = base
    ...    def __call__(self, state): return state['stepsize']<self.base
    ...    def __str__(self): return "stepsize<%g"%self.base
    >>> converge = stepsize(0.001)
    >>> diverge = iters(100)
    >>> result,why = converge.status(dict(stepsize=21.2,iters=20))
    >>> print("%s %s"%(result, ", ".join(str(c) for c in why)))
    False not stepsize<0.001
    >>> result,why = diverge.status(dict(stepsize=21.2,iters=129))
    >>> print("%s %s"%(result, ", ".join(str(c) for c in why)))
    True iters>100

Note that status will be less efficient than direct evaluation
because it has to test all branches and construct the result list.
Normally the And and Or calculations an short circuit, and only
compute what they need to guarantee the resulting truth value.
"""


class Condition(object):
    """
    Condition abstract base class.

    Subclasses should define __call__(self, *args, **kw)
    which returns True if the condition is satisfied, and
    False otherwise.
    """

    def __and__(self, condition):
        return And(self, condition)

    def __or__(self, condition):
        return Or(self, condition)

    def __xor__(self, condition):
        return Xor(self, condition)

    def __invert__(self):
        return Not(self)

    def __call__(self, *args, **kw):
        raise NotImplementedError

    def _negate(self):
        return _Bar(self)

    def status(self, *args, **kw):
        """
        Evaluate the condition, returning both the status and a list of
        conditions.  If the status is true, then the conditions are those
        that contribute to the true status.  If the status is false, then
        the conditions are those that contribute to the false status.
        """
        stat = self.__call__(*args, **kw)
        if stat:
            return stat, [self]
        else:
            return stat, [Not(self)]

    def primitives(self):
        """
        Return a list of terms in the condition.
        """
        return set([self])


class And(Condition):
    """
    True if both conditions are satisfied.
    """

    def __init__(self, left, right):
        self.left, self.right = left, right

    def __call__(self, *args, **kw):
        return self.left(*args, **kw) and self.right(*args, **kw)

    def __str__(self):
        return "(%s and %s)" % (self.left, self.right)

    def _negate(self):
        return Or(self.left._negate(), self.right._negate())

    def status(self, *args, **kw):
        lstat, lcond = self.left.status(*args, **kw)
        rstat, rcond = self.right.status(*args, **kw)
        if lstat and rstat:
            return True, lcond + rcond
        elif lstat:
            return False, rcond
        elif rstat:
            return False, lcond
        else:
            return False, lcond + rcond

    def primitives(self):
        return self.left.primitives() | self.right.primitives()


class Or(Condition):
    """
    True if either condition is satisfied
    """

    def __init__(self, left, right):
        self.left, self.right = left, right

    def __call__(self, *args, **kw):
        return self.left(*args, **kw) or self.right(*args, **kw)

    def __str__(self):
        return "(%s or %s)" % (self.left, self.right)

    def _negate(self):
        return And(self.left._negate(), self.right._negate())

    def status(self, *args, **kw):
        lstat, lcond = self.left.status(*args, **kw)
        rstat, rcond = self.right.status(*args, **kw)
        if lstat and rstat:
            return True, lcond + rcond
        elif lstat:
            return True, lcond
        elif rstat:
            return True, rcond
        else:
            return False, lcond + rcond

    def primitives(self):
        return self.left.primitives() | self.right.primitives()


class Xor(Condition):
    """
    True if only one condition is satisfied
    """

    def __init__(self, left, right):
        self.left, self.right = left, right

    def __call__(self, *args, **kw):
        l, r = self.left(*args, **kw), self.right(*args, **kw)
        return (l or r) and not (l and r)

    def __str__(self):
        return "(%s xor %s)" % (self.left, self.right)

    def _negate(self):
        return Xor(self.left, self.right._negate())

    def status(self, *args, **kw):
        lstat, lcond = self.left.status(*args, **kw)
        rstat, rcond = self.right.status(*args, **kw)
        if lstat ^ rstat:
            return True, lcond + rcond
        else:
            return False, lcond + rcond

    def primitives(self):
        return self.left.primitives() | self.right.primitives()


class Not(Condition):
    """
    True if condition is not satisfied
    """

    def __init__(self, condition):
        self.condition = condition

    def __call__(self, *args, **kw):
        return not self.condition(*args, **kw)

    def __str__(self):
        return "not " + str(self.condition)

    def _negate(self):
        return self.condition

    def status(self, *args, **kw):
        stat, cond = self.condition.status()
        return not stat, cond

    def __eq__(self, other):
        return isinstance(other, Not) and self.condition == other.condition

    def __ne__(self, other):
        return not isinstance(other, Not) or self.condition != other.condition

    def primitives(self):
        return self.condition.primitives()


class _Bar(Condition):
    """
    This is an internal condition structure created solely to handle
    negated primitives when computing status.  It should not be used
    externally.
    """

    def __init__(self, condition):
        self.condition = condition

    def __call__(self, *args, **kw):
        return not self.condition(*args, **kw)

    def _negate(self):
        return self.condition

    def status(self, *args, **kw):
        stat, cond = self.condition.status(*args, **kw)
        return not stat, cond

    def __str__(self):
        return "not " + str(self.condition)

    def primitives(self):
        return self.condition.primitives()


class Constant(Condition):
    """
    Constants true and false.
    """

    def __init__(self, value):
        self._value = value

    def __call__(self, *args, **kw):
        return self._value

    def __str__(self):
        return str(self._value)


true = Constant(True)
false = Constant(False)
