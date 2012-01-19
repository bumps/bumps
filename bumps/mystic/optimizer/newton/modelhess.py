'''

Created on Nov 23, 2009

@author: Ismet Sahin and Christopher Meeting

% NOTES:
%    Currently we are not implementing steps 1, 14, and 15 (TODO)

% This function performs perturbed Cholesky decomposition (CD) as if the input
% Hessian matrix is positive definite.  The code for perturbed CD resides in
% choldecomp.m file which returns the factored lower triangle matrix L and a
% number, maxadd, specifying the largest number added to a diagonal element of H
% during the CD decomposition.  This function checks if the decomposition is
% completed without adding any positive number to the diagonal elements of H,
% i.e. maxadd <= 0.  Otherwise, this function adds the least number to the
% diagonals of H which makes it positive definite based on maxadd and other
% entries in H.

%         A1 =[2     0    2.4
%               0     2     0
%              2.4     0     3]
%
%         A2 =[2     0    2.5
%               0     2     0
%              2.5     0     3]
%
%         A3 =[2     0    10
%               0     2     0
%              10     0     3]
'''

from numpy import sqrt, diag, Inf, ones, array
import choldecomp as chol

def modelhess(n, H, s, macheps):

    # STEP I.
    sqrteps = sqrt(macheps)

    #2-4.
    H_diag = diag(H)
    maxdiag = max(H_diag)
    mindiag = min(H_diag)

    # 5.
    maxposdiag = max(0, maxdiag)

    # 6. mu is the amount to be added to diagonal of H before the Cholesky decomp.
    # If the minimum diagonal is much much smaller than the maximum diagonal element
    # then adjust mu accordingly otherwise mu = 0.
    if mindiag <= sqrteps * maxposdiag:
        mu = 2 * (maxposdiag - mindiag) * sqrteps - mindiag
        maxdiag = maxdiag + mu
    else:
        mu = 0

    # 7. maximum of off-diagonal elements of H
    diag_infinite = diag(Inf * ones(n))
    maxoff = (H - diag_infinite).max()

    # 8. if maximum off diagonal element is much much larger than the maximum
    # diagonal element of the Hessian H
    if maxoff * (1 + 2 * sqrteps) > maxdiag:
        mu = mu + (maxoff - maxdiag) + 2 * sqrteps * maxoff
        maxdiag = maxoff * (1 + 2 * sqrteps)

    # 9.
    if maxdiag == 0:
        mu = 1
        maxdiag = 1

    # 10. mu>0 => need to add mu amount to the diagonal elements: H = H + mu*I
    if mu > 0:
        diag_mu = diag(mu * ones(n))
        H = H + diag_mu

    # 11.
    maxoffl = sqrt(max(maxdiag, maxoff/n))

    # STEP II. Perform perturbed Cholesky decomposition H + D = LL' where D is a
    # diagonal matrix which is implicitly added to H during decomposition with some
    # positive elements if H is not positive definite. The output variable maxadd
    # indicates the maximum number added to a diagonal entry of the Hesian, i.e. the
    # maximum of D.
    # If maxadd is returned 0, then H was indeed pd and L is the resulting factor.
    # 12.
    L, maxadd = chol.choldecomp(n, H, maxoffl, macheps)

    # STEP III.
    # 13. If maxadd <= 0, we are done H was positive definite.
    if maxadd > 0:
        # H wasnot positive definite
        print 'WARNING: Hessian is not pd. Max number added to H is ', maxadd, '\n'
        maxev = H[0,0]
        minev = H[0,0]
        for i in range(1,n+1):
            offrow = sum(abs(H[0:i-1, i-1])) + sum(abs(H[i-1, i:n]))
            maxev = max(maxev, H[i-1,i-1] + offrow)
            minev = min(minev, H[i-1,i-1] - offrow)

        sdd = (maxev - minev) * sqrteps - minev
        sdd = max(sdd, 0)
        mu = min(maxadd, sdd)
        H = H + diag(mu * ones(n))
        L, maxadd = chol.choldecomp(n, H, 0, macheps)

    return L,H

def example_call():
    A1 = array([[2, 0, 2.4],[0, 2, 0],[2.4, 0, 3]])
    L, H = modelhess(3, A1, 0, 1e-16)
    print 'Lower matrix :\n', L
    print '\n Hessian :\n', H
    print '-------------------'
    A2 = array([[2, 0, 2.5],[0, 2, 0],[2.5, 0, 3]])
    L, H = modelhess(3, A2, 0, 1e-16)
    print 'Lower matrix :\n', L
    print '\n Hessian :\n', H
    print '-------------------'
    A3 = array([[2, 0, 10],[0, 2, 0],[10, 0, 3]])
    L, H = modelhess(3, A3, 1, 1e-16)
    print 'Lower matrix :\n', L
    print '\n Hessian :\n', H

def main():
    example_call();

if __name__ == "__main__":
    main();
