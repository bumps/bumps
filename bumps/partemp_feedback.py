# This program is public domain
# Author: Paul Kienzle
"""
Parallel tempering for continuous function optimization and uncertainty analysis.

The program performs Markov chain Monte Carlo exploration of a probability
density function using a combination of random and differential evolution
updates.
"""
from __future__ import division, print_function

__all__ = ["parallel_tempering_feedback"]

import numpy as np
from numpy import asarray, zeros, ones, exp, diff, std, inf, \
    array, nonzero, sqrt, zeros_like
from numpy.linalg import norm
from numpy.random import rand, randn, randint, permutation
from scipy.stats import linregress

def every_ten(step, x, fx, P, E):
    if step % 10:
        print(step, fx[step], x[step])

def find_freq(temp, direction, histogram, f):
    '''
    Determines the fraction of chains which has visited a certain extreme.
    
    :Parameters:
    
    *Temp* : int
        The index of the specific temperature at which you are evalutating at.
    
    *Direction* : int
        The extreme that the particles are diffusing to.
        *0* for up
        *1* for down
        
    *Histogram* : array
        The histogram object of the current simulation
    '''
    try:
        value = histogram[temp][direction]/sum(histogram[temp])
    except RuntimeWarning:
        pass
    if np.isnan(value):
        value = 0
    f.write("value: " + str(value) + " top: " + str(histogram[temp][direction]) + " bottom: " + str(sum(histogram[temp])) + "\n")
    return value

def update_histogram(histogram, labels):
    for i in range(len(labels)):
        if labels[i] == 1:
            histogram[i][0] = histogram[i][0] + 1
        elif labels[i] == 2:
            histogram[i][1] = histogram[i][1] + 1
    return histogram

def update_labels(labels):
    labels[0] = 1
    labels[1] = 1
    labels[-1] = 2
    labels[-2] = 2
    return labels

def find_dif(T, histogram, f):
    freq = []
    for i in range(0, len(T)):
        freq.append(find_freq(i, 0, histogram, f))
    slope, intercept, r_value, p_value, std_err = linregress(np.linspace(0, len(freq) - 1, len(freq)), freq)
    return slope

def find_constant(T, dif):
    total = 0
    for i in range(len(T) - 1):
        total += prob_dist(i, T, dif)*(T[i + 1] - T[i])
        # print(total)
    return 1/total


def prob_dist(index, T, dif):
    return sqrt(abs(1/(T[index+1] - T[index]) * dif))

def optimize_temperature(T, constant, dif):
    new_temperature = [T[0]]
    index = 0
    target = 2/len(T)
    current_interval = constant*prob_dist(index, T, dif)
    rect = current_interval*(T[index+1] - T[index])
    for i in range(1, len(T)-1):
        while 1:
            if rect < target:
                target -= rect
                index += 1
                current_interval = constant*prob_dist(index, T, dif)
                rect = current_interval*(T[index + 1] - T[index])
            else:
                new_temp = target/current_interval + new_temperature[i - 1]
                new_temperature.append(new_temp)
                rect -= target
                target = ((i+2)/len(T)) - ((i+1)/len(T))
                break
    new_temperature.append(T[-1])
    return asarray(new_temperature)


def parallel_tempering_feedback(nllf, p, bounds, T=None, steps=1000,
                       CR=0.9, burn=1000,
                       monitor=every_ten,
                       logfile=None,
                       labels=None):
    f = open("data.txt","w")
    r"""
    Perform a MCMC walk using multiple temperatures in parallel.

    :Parameters:

    *nllf* : function(vector) -> float
        Negative log likelihood function to be minimized.  $\chi^2/2$ is a
        good choice for curve fitting with no prior restraints on the possible
        input parameters.
    *p* : vector
        Initial value
    *bounds* : vector, vector
        Box constraints on the parameter values.  No support for indefinite
        or semi-definite programming at present
    *T* : vector | 0 < T[0] < T[1] < ...
        Temperature vector.  Something like logspace(-1,1,10) will give
        you 10 logarithmically spaced temperatures between 0.1 and 10.  The
        maximum temperature T[-1] determines the size of the barriers that
        can be easily jumped.  Note that the number of temperature values
        limits the amount of parallelism available in the algorithm, so it
        may gather statistics more quickly, though it will not necessarily
        converge any faster.
    *steps* = 1000 : int
        Length of the accumulation vector.  The returned history will store
        this many values for each temperature.  These values can be used in
        a weighted histogram to determine parameter uncertainty.
    *burn* = 1000 : int | [0,inf)
        Number of iterations to perform in addition to steps.  Only the
        last *steps* points will be preserved for each temperature.  Since
        the
        value should be in the same order as *steps* to be sure that the
        full history is acquired.
    *CR* = 0.9 : float | [0,1]
        Cross-over ratio.  This is the differential evolution crossover
        ratio to use when computing step size and direction.  Use a small
        value to step through the dimensions one at a time, or a large value
        to step through all at once.
    *monitor* = every_ten : function(int step, vector x, float fx) -> None
        Function to called at every iteration with the step number the
        best point and the best value.
    *logfile* = None : string
        Name of the file which will log the history of every accepted step.
        Note that this includes all of the burn steps, so it can get very
        large.

    :Returns:

    *history* : History
        Structure containing *best*, *best_point* and *buffer*.  *best* is
        the best nllf value seen and *best_point* is the parameter vector
        which yielded *best*.  The list *buffer* contains lists of tuples
        (step, temperature, nllf, x) for each temperature.
    """
    N = len(T)  # Number of temperatures
    history = History(logfile=logfile, streams=N, size=steps, burn=burn, var_labels=labels)
    bounder = ReflectBounds(*bounds)
    #stepper = RandStepper(bounds, tol=0.2/T[-1])
    stepper = Stepper(bounds, history)
    dT = diff(1. / asarray(T)) # Difference in temperatures
    P = asarray([p] * N)   # Points
    E = ones(N) * nllf(p)  # Values   
    labels = update_labels(zeros(N)) # 0 neither, 1 up, 2 down
    swap_history = asarray(np.linspace(0, N - 1, N))
    histograms = np.array([[0,0] for i in range(N)]) #0 up, 1 down
    
    history.save(step=0, temperature=T, energy=E, point=P, swap_history=swap_history, labels=labels)
    total_accept = zeros(N) # Histogram of accept
    total_swap = zeros(N - 1) # Histogram of swap
    swap_increment = 100
    swap_step = swap_increment
    for step in range(1, steps + burn):
        if step == swap_step:
            f.write("Step: " + str(step) + "\n")
            f.write("Histograms \n" + str(histograms) + "\n")
            f.write("Temp \n" + str(T) + "\n")

            dif = find_dif(T, histograms, f)
            constant = find_constant(T, dif)
            T = optimize_temperature(T, constant, dif)

            f.write("constant: " + str(constant) + "\n")
            f.write("dif: " + str(dif) + "\n")
            f.write("New temp \n" + str(T) + "\n")

            dT = diff(1. / asarray(T))
            histograms = np.array([[0,0] for i in range(N)])
            labels = update_labels(zeros(N))

            swap_step += swap_increment

#       Take a step
        R = rand()
        '''
        def jiggle(self, p, noise):
        delta = randn(len(p)) * self.step * noise
        assert norm(delta) != 0
        return p + delta
        '''
        if step < 20 or R < 0.2:
            #action = 'jiggle'
            Pnext = [stepper.jiggle(p, 0.01 * t / T[-1]) for p, t in zip(P, T)]
        elif R < 0.4:
            #action = 'direct'
            Pnext = [stepper.direct(p, i) for i, p in enumerate(P)]
        else:
            #action = 'diffev'
            Pnext = [stepper.diffev(p, i, CR=CR) for i, p in enumerate(P)]

        # Test constraints
        Pnext = asarray([bounder.apply(p) for p in Pnext])
        
        # Temperature dependent Metropolis update
        Enext = asarray([nllf(p) for p in Pnext])
        accept = exp(-(Enext - E) / T) > rand(N)
        # print step,action
        # print "dP"," ".join("%.6f"%norm((pn-p)/stepper.step) for pn,p in zip(P,Pnext))
        # print "dE"," ".join("%.1f"%(en-e) for en,e in zip(E,Enext))
        # print "En"," ".join("%.1f"%e for e in Enext)
        # print "accept",accept
        E[accept] = Enext[accept]
        P[accept] = Pnext[accept]
        total_accept += accept
        # print("point: \n", p)
        # Accumulate history for population based methods
        history.save(step, temperature=T, energy=E, point=P, swap_history=swap_history, labels=labels, changed=accept)

        # print "best",history.best

        # Swap chains across temperatures
        # Note that we are are shuffling from high to low so that if a good
        # point is found at a high temperature which push it immediately as
        # low as we can go rather than risk losing it at the next high temp
        # step.
        if step % 1 == 0:
            swap = zeros(N - 1)
            for i in range(N - 2, -1, -1):
                if exp((E[i + 1] - E[i]) * dT[i]) > rand():
                    swap[i] = 1
                    # switch the energy states around
                    swap_history[i + 1], swap_history[i] = swap_history[i], swap_history[i + 1]
                    labels[i + 1], labels[i] = labels[i], labels[i + 1]
                    E[i + 1], E[i] = E[i], E[i + 1]
                    P[i + 1], P[i] = P[i] + 0, P[i + 1] + 0
            total_swap += swap
            labels = update_labels(labels)
            histograms = update_histogram(histograms, labels)
            #assert nllf(P[0]) == E[0]

        # Monitoring
        monitor(step, history.best_point, history.best, P, E)
        interval = 100
        if 0 and step % interval == 0:
            print("max r",
                  max(["%.1f" % norm(p - P[0]) for p in P[1:]]))
            # print "min AR",argmin(total_accept),min(total_accept)
            # print "min SR",argmin(total_swap),min(total_swap)
            print("AR", total_accept)
            print("SR", total_swap)
            print("s(d)", [int(std([p[i] for p in P]))
                           for i in (3, 7, 11, -1)])
            total_accept *= 0
            total_swap *= 0
            
    f.close()
    return history


class History(object):

    def __init__(self, streams=None, size=1000, logfile=None, burn=0, var_labels=None):
        # Allocate buffers
        self.size = size
        self.buffer = [[] for _ in range(streams)]
        # Prepare log file
        if logfile is not None:
            self.log = open(logfile, 'w')
            print("# Step Temperature Energy Point", file=self.log)
        else:
            self.log = None
        # Track the optimum
        self.best = inf
        self.burn = burn
        self.temperatures, self.points, self.energies, self.swap, self.norm_temperatures = [], [], [], [], []
        self.var_labels = var_labels
        self.test = []

    def save(self, step, temperature, energy, point, swap_history, labels, changed=None):
        if step > self.burn:
            self.temperatures.append(temperature)
            self.norm_temperatures.append(temperature[swap_history.argsort()])
            self.points.append(point[swap_history.argsort()])
            self.energies.append(energy[swap_history.argsort()])
            self.swap.append(swap_history.argsort())
            self.test.append(step)

        if changed is None:
            changed = ones(len(temperature), 'b')
        for i, a in enumerate(changed):
            if a:
                self._save_point(
                    step, i, temperature[i], energy[i], point[i] + 0, labels)

    def _save_point(self, step, i, T, E, P, labels):
        # Save in buffer
        S = self.buffer[i]
        if len(S) >= self.size:
            S.pop(0)
        if len(S) > 0:
            # print "P",P
            # print "S",S[-1][3]
            assert norm(P - S[-1][3]) != 0
        S.append((step, T, E, P, labels))
        # print "stream",i,"now len",len(S)
        # Track of the optimum
        if E < self.best:
            self.best = E
            self.best_point = P
        # Log to file
        if self.log:
            point_str = " [" + "".join("%.6g" % v for v in P) + "]"
            label_str = " [" + " ".join("%.6g" % v for v in labels) + "]"
            print(step, T, E, point_str, label_str, file=self.log)
            self.log.flush()

    def draw(self, stream, k):
        """
        Return a list of k items drawn from the given stream.

        If the stream is too short, fewer than n items may be returned.
        """
        S = self.buffer[stream]
        n = len(S)
        return [S[i] for i in choose(n, k)] if n > k else S[:]

    def toString(self):
        return self.buffer

    def plot(self, output_file=None):
        from pylab import plot, figure, semilogy, ylabel, xlabel, suptitle, gca, hist, subplot, legend, cm
        from dream import corrplot, views, stats

        # # Temperature History
        # figure()
        # x = np.linspace(0, len(self.temperatures) - 1, len(self.temperatures))
        # for i in range(len(self.temperatures[0])):
        #     semilogy(x, map(lambda t: t[i], self.temperatures), hold=True)
        # ylabel("Temperature")
        # xlabel("Generation")
        # suptitle("Temperature History")

        # Parameter History
        figure()
        gca().set_color_cycle([cm.gist_ncar(i) for i in np.linspace(0, 0.9, len(self.points[0][0]))])
        x = np.linspace(0, len(self.points) - 1, len(self.points))
        graphs = []
        for i in range(len(self.points[0])):
            for j in range(len(self.points[0][0])-1, -1, -1):
                graph, = plot(x, map(lambda p:p[i][j], self.points), hold=True, marker='.', markersize=1)
                graphs.append(graph)
        group = [tuple(graphs[::i+1]) for i in range(len(self.points[0][0]))]
        legend(group, self.var_labels, markerscale=10)
        ylabel("Value")
        xlabel("Generation")
        suptitle("Parameter History")

        # Energy History
        figure()
        # x = np.linspace(0, len(self.energies) - 1, len(self.energies))
        for i in range(len(self.energies[0])):
            plot(x, map(lambda e:e[i]*-1, self.energies), ',', hold=True)
        ylabel("Value")
        xlabel("Generation")
        suptitle("Log Likelihood")

        # Swap History
        figure()
        for i in range(len(self.swap[0])):
            semilogy(x, map(lambda s,t:t[s[i]], self.swap, self.temperatures), hold=True)
        suptitle("Temperature Swap History")
        ylabel("Temperature")
        xlabel("Generation")

        # 1D histograms
        figure()
        nw, nh = views.tile_axes(len(self.points[0][0]))
        weights = (1./asarray(self.norm_temperatures)).flatten()
        cbar = views._make_fig_colorbar(asarray(self.energies).flatten() * -1)
        ONE_SIGMA = 1 - 2*0.15865525393145705
        parameters = []

        f = open("Hello.txt", "w")
        for i in range(len(self.points[0][0])):
            parameter = []
            map(lambda p:parameter.extend([t[i] for t in p][::-1]), self.points)
            parameters.append(parameter)

            subplot(nw, nh, i+1)
            p95, p68, p0 = stats.credible_intervals(x=asarray(parameter), weights=weights, ci=[0.95, ONE_SIGMA, 0.0])
            f.write("95 \n" + str(p95) + "\n 68 \n" + str(p68) + "\n  0 \n" + str(p0))
            mean, std = stats.stats(x=asarray(parameter), weights=weights)
            views._make_logp_histogram(asarray(parameter), asarray(self.energies).flatten() * -1, 30, p95, weights, cbar)
            views._decorate_histogram(stats.VarStats(label=self.var_labels[i], index=i+1, p95=p95, p68=p68,
                                      median=p0[0], mean=mean, std=std, best=self.best_point[i]))

        suptitle("1/T")

        # 1D histograms
        figure()
        nw, nh = views.tile_axes(len(self.points[0][0]))
        weights = (asarray(self.norm_temperatures)).flatten()
        cbar = views._make_fig_colorbar(asarray(self.energies).flatten() * -1)
        parameters = []

        for i in range(len(self.points[0][0])):
            parameter = []
            map(lambda p:parameter.extend([t[i] for t in p][::-1]), self.points)
            parameters.append(parameter)

            subplot(nw, nh, i+1)
            p95, p68, p0 = stats.credible_intervals(x=asarray(parameter), weights=weights, ci=[0.95, ONE_SIGMA, 0.0])
            mean, std = stats.stats(x=asarray(parameter), weights=weights)
            views._make_logp_histogram(asarray(parameter), asarray(self.energies).flatten() * -1, 30, p95, weights, cbar)
            views._decorate_histogram(stats.VarStats(label=self.var_labels[i], index=i+1, p95=p95, p68=p68,
                                      median=p0[0], mean=mean, std=std, best=self.best_point[i]))
        suptitle("T")

        # 1D histograms
        figure()
        nw, nh = views.tile_axes(len(self.points[0][0]))
        parameters = []
        for i in range(len(self.points[0][0])):
            parameter = []
            map(lambda p:parameter.extend([t[i] for t in p]), self.points)
            parameters.append(parameter)

            subplot(nw, nh, i+1)
            hist(parameter, 50, normed=1)
            frame = gca()
            frame.axes.get_yaxis().set_ticks([])

        # 2D histograms
        figure()
        corrplot.Corr2d(parameters, bins=50, labels=self.var_labels).plot()
        # for i in range(len(parameters) - 1):
        #     for j in range(i + 1, len(parameters)):
        #         subplot(len(parameters)-1, len(parameters)-1, i*(len(parameters)-1)+j)
        #         hist2d(parameters[i], parameters[j], num_bins)
        #         frame = gca()
        #         frame.axes.get_yaxis().set_ticks([])
        #         frame.axes.get_xaxis().set_ticks([])


class Stepper(object):

    def __init__(self, bounds, history):
        low, high = bounds
        self.offset = low
        self.step = (high - low)
        self.history = history

    def diffev(self, p, stream, CR=0.8, noise=0.05):
        if len(self.history.buffer[stream]) < 20:
            # print "jiggling",stream,stream,len(self.history.buffer[stream])
            return self.jiggle(p, 1e-6)
        # Ideas incorporated from DREAM by Vrugt
        N = len(p)
        # Select to number of vector pair differences to use in update
        # using k ~ discrete U[1,max pairs]
        k = randint(4) + 1

        # Select 2*k members at random
        parents = [v[3] for v in self.history.draw(stream, 2 * k)]
        k = len(parents) // 2  # May not have enough parents
        pop = array(parents)
        # print "k",k
        # print "parents",parents
        # print "pop",pop

        # Select the dims to update based on the crossover ratio, making
        # sure at least one significant dim is selected
        while True:
            vars = nonzero(rand(N) < CR)
            if len(vars) == 0:
                vars = [randint(N)]
            step = np.sum(pop[:k] - pop[k:], axis=0)
            if norm(step[vars]) > 0:
                break

        # Weight the size of the jump inversely proportional to the
        # number of contributions, both from the parameters being
        # updated and from the population defining the step direction.
        gamma = 2.38 / sqrt(2 * len(vars) * k)

        # Apply that step with F scaling and noise
        eps = 1 + noise * (2 * rand(N) - 1)
        # print "j",j
        # print "gamma",gamma
        # print "step",step.shape
        # print "vars",vars.shape
        delta = zeros_like(p)
        delta[vars] = gamma * (eps * step)[vars]
        assert norm(delta) != 0
        return p + delta

    def direct(self, p, stream):
        if len(self.history.buffer[stream]) < 20:
            # print "jiggling",stream,len(self.history.buffer[stream])
            return self.jiggle(p, 1e-6)
        pair = self.history.draw(stream, 2)
        delta = pair[0][3] - pair[1][3]
        if norm(delta) == 0:
            print("direct should never return identical points!!")
            return self.random(p)
        assert norm(delta) != 0
        return p + delta

    def jiggle(self, p, noise):
        delta = randn(len(p)) * self.step * noise
        assert norm(delta) != 0
        return p + delta

    def random(self, p):
        delta = rand(len(p)) * self.step + self.offset
        assert norm(delta) != 0
        return p + delta

    def subspace_jiggle(self, p, noise, k):
        n = len(self.step)
        if n < k:
            idx = slice(None)
            k = n
        else:
            idx = choose(n, k)
        delta = zeros_like(p)
        delta[idx] = randn(k) * self.step[idx] * noise
        assert norm(delta) != 0
        return p + delta


class ReflectBounds(object):
    """
    Reflect parameter values into bounded region
    """

    def __init__(self, low, high):
        self.low, self.high = [asarray(v, 'd') for v in (low, high)]

    def apply(self, y):
        """
        Update x so all values lie within bounds

        Returns x for convenience.  E.g., y = bounds.apply(x+0)
        """
        minn, maxn = self.low, self.high
        # Reflect points which are out of bounds
        idx = y < minn
        y[idx] = 2 * minn[idx] - y[idx]
        idx = y > maxn
        y[idx] = 2 * maxn[idx] - y[idx]

        # Randomize points which are still out of bounds
        idx = (y < minn) | (y > maxn)
        y[idx] = minn[idx] + rand(sum(idx)) * (maxn[idx] - minn[idx])
        return y


def choose(n, k):
    """
    Return an array of k things selected from a pool of n without replacement.
    """
    # At k == n/4, need to draw an extra 15% to get k unique draws
    if k > n / 4 or n < 100:
        idx = permutation(n)[:k]
    else:
        s = set(randint(n, size=k))
        while len(s) < k:
            s.add(randint(n))
        idx = array([si for si in s])
    if len(set(idx)) != len(idx):
        print("choose(n,k) contains dups!!", n, k)
    return idx
