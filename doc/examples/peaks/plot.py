import sys
import json

import numpy as np
import pylab

def plot(X,Y,theory,data,err):
    #print "theory",theory[1:6,1:6]
    #print "data",data[1:6,1:6]
    #print "delta",(data-theory)[1:6,1:6]
    pylab.subplot(3,1,1)
    pylab.pcolormesh(X,Y, data)
    pylab.subplot(3,1,2)
    pylab.pcolormesh(X,Y, theory)
    pylab.subplot(3,1,3)
    pylab.pcolormesh(X,Y, (data-theory)/(err+1))

def load_results(filename):
    """
    Reload results from the json file created by Peaks.save
    """
    data = json.load(open(filename))
    # Convert array info back into numpy arrays
    data.update( (k,np.array(data[k]))
                 for k in ('X', 'Y', 'data', 'err', 'theory') )
    return data

def main():
    data = load_results(sys.argv[1])
    plot(data['X'],data['Y'],data['theory'],data['data'],data['err'])
    pylab.show()

if __name__ == "__main__": main()
