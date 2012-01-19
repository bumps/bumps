'''

Created on Nov 23, 2009

@author: Ismet Sahin and Christopher Meeting
'''
from numpy import sqrt, diag, zeros, array

def choldecomp(n, H, maxoffl, macheps):
    minl = (macheps)**(0.25) * maxoffl

    if maxoffl == 0:
        # H is known to be a positive definite matrix
        maxoffl = sqrt(max(abs(diag(H))))

    minl2 = sqrt(macheps) * maxoffl

    # 3. maxadd is the number (R) specifying the maximum amount added to any
    # diagonal entry of Hessian matrix H
    maxadd = 0

    # 4. form column j of L
    L = zeros((n,n))
    for j in range(1,n+1):
        L[j-1,j-1] = H[j-1,j-1] - sum(L[j-1, 0:j-1]**2)
        minljj = 0
        for i in range(j+1, n+1):
            L[i-1,j-1] = H[j-1,i-1] - sum(L[i-1, 0:j-1] * L[j-1, 0:j-1])
            minljj = max(abs(L[i-1,j-1]), minljj)

        # 4.4
        minljj = max(minljj/maxoffl, minl)

        # 4.5
        if L[j-1,j-1] > minljj**2:
            # normal Cholesky iteration
            L[j-1,j-1] = sqrt(L[j-1,j-1])
        else:
            # augment H[j-1,j-1]
            if minljj < minl2:
                minljj = minl2    # occurs only if maxoffl = 0

            maxadd = max(maxadd, minljj**2 - L[j-1,j-1])
            L[j-1,j-1] = minljj

        # 4.6
        L[j:n, j-1] = L[j:n, j-1] / L[j-1,j-1]


    return L, maxadd

def example_call():
    A1 = array([[2, 0, 2.4],[0, 2, 0],[2.4, 0, 3]])
    L, maxadd = choldecomp(3, A1, 0, 1e-16)
    print 'Lower matrix L\n', L
    print '\n The number ', maxadd, ' is added to diagonal entries of the Hesian.'


#example_call()
