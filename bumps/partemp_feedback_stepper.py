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
from pylab import plot, figure, suptitle, legend, semilogx
import math
def every_ten(step, x, fx, P, E):
    if step % 10:
        print(step, fx[step], x[step])

def find_freq(temp, direction, histogram):
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
    value = histogram[temp][direction]/sum(histogram[temp])
    if np.isnan(value):
        value = 0
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

def find_dif(T, histogram):
    freq = []
    for i in range(0, len(T)):
        freq.append(find_freq(i, 0, histogram))
    slope, intercept, r_value, p_value, std_err = linregress(np.linspace(0, len(freq) - 1, len(freq)), freq)
    return slope

def find_constant(T, dif):
    total = 0
    for i in range(len(T) - 1):
        total += prob_dist(i, T, dif)*(T[i + 1] - T[i])
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
                       labels=None, mapper=None):
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

    swap_frequency = zeros(N)
    acceptance_frequency = zeros(N)
    step_size = zeros(N)
    energy_change = zeros(N)
    figure()
    scale = 1
    directScale = asarray([1.,1.,1.])
    scale_history = [scale]
    count = 1
    accept_optimize = 300
    swap_increment = accept_optimize*2
    temperature_optimize = swap_increment
    acceptance_history = []
    step_history = []
    avg_accept = []
    percentage=0
    formerBest = min(E)
    difference = 0
    reset = True
    averageEn = [difference]
    part = 0.5
    for step in range(1, steps + burn):
        if step == (steps+burn - 1):
            suptitle("Acceptance")
            legend()
            figure()
            plot(step_history, acceptance_history, ".")
            suptitle("Median Acceptance")
            figure()
            plot(step_history, avg_accept, ".")
            suptitle("Average Acceptance")
            figure()
            plot(T, step_size/step, '.')
            suptitle("Step")
            figure()
            plot(T, energy_change/step, '.')
            suptitle("Energy")
            figure()
            plot(np.linspace(0, len(scale_history) - 1, len(scale_history)), scale_history, '.')
            suptitle("Scale")
            figure()

        if step == accept_optimize:
            # Scale optimizer
            # print(acceptance_frequency)
            # print ("med", np.median(acceptance_frequency/swap_increment))
            if np.average(acceptance_frequency/swap_increment) != 0:
                scale *= .234/np.average(acceptance_frequency/swap_increment)
            scale_history.append(scale)
            semilogx(T, acceptance_frequency/swap_increment, hold=True, label=str(count), marker=',')
            acceptance_history.append(np.median(acceptance_frequency/swap_increment))
            avg_accept.append(np.average(acceptance_frequency/swap_increment))
            step_history.append(step)

            #switch between steppers
            directScale[percentage] = scale

            if (formerBest - history.best) <= 10 and difference <= 0:
                if percentage == 2:
                    print("yes")
                    print(part)
                    part = min(N, part + .5)
                    print(part)
                print("Reset")
                percentage = 2
                print(part)
            elif percentage == 2:
                percentage = 0

            if (formerBest - history.best) <= min(difference,150) and percentage != 2:
                print("Swap")
                percentage = 1 - percentage

            scale = directScale[percentage]


            difference = formerBest - history.best
            averageEn.append(difference)
            formerBest = history.best

            # Max temperature
            # T[-1] = T[-2] + (T[-1] - T[-2]) * min(2, .4/(acceptance_frequency[-1]/swap_increment))
            # T[0] = min(abs(T[1] - 0.01), T[0] * .1/(acceptance_frequency[0]/swap_increment))

            # T = np.logspace(log(T[0], 10), log(T[-1], 10), N)
            # print(T[0])
            # print(T[1])
            # Reset graphs
            histograms = np.array([[0,0] for i in range(N)])
            labels = update_labels(zeros(N))

            acceptance_frequency = zeros(N)
            swap_frequency = zeros(N)


            # Increment
            accept_optimize += swap_increment
            count += 1

        # if False:
        if step == temperature_optimize:
            # Feedback exchange
            dif = find_dif(T, histograms)
            constant = find_constant(T, dif)
            T = optimize_temperature(T, constant, dif)
            dT = diff(1. / asarray(T))

            # Reset Graphs
            acceptance_frequency = zeros(N)
            swap_frequency = zeros(N)

            # Increment
            temperature_optimize += swap_increment

#       Take a step
#         R = rand()
        if step < 20 or percentage:
            # print("jiggle")
            delta = [stepper.jiggle(p, 0.01 * t / T[-1]) for p, t in zip(P, T)]
        else:
            # print("de")
            delta = [stepper.diffev(p, i, CR=CR) for i, p in enumerate(P)]

        if percentage == 2:
            # delta = [stepper.direct(p, i) for i, p in enumerate(P)]
            delta = [stepper.subspace_jiggle(p, 0.01 * t / T[-1], 3) for p,t in zip(P,T)]


        # delta = [stepper.jiggle(p, 0.01 * (t / T[-1])) for p, t in zip(P, T)]

        # delta = [stepper.diffev(p, i, CR=CR) for i, p in enumerate(P)]

        # if step < 20 or R < 0.4:
        #     action = 'jiggle'
        #     delta = [stepper.jiggle(p, 0.01 * t / T[-1]) for p, t in zip(P, T)]
        # elif R < 0.6:
        #     action = 'direct'
        #     delta = [stepper.direct(p, i) for i, p in enumerate(P)]
        # else:
        #     action = 'diffev'
        #     delta = [stepper.diffev(p, i, CR=CR) for i, p in enumerate(P)]

        # Test constraints
        Pnext = P + asarray(delta)/scale
        Pnext = asarray([bounder.apply(p) for p in Pnext])
        if(step > burn):
            #print("Min", Pnext[0] - P[0])
            #print("Max", Pnext[-1] - P[-1])
            pass

        # Temperature dependent Metropolis update
        # Enext = asarray([nllf(p) for p in Pnext])
        Enext = asarray(mapper(Pnext))
        accept = exp(-(Enext - E) / T) > rand(N)
        # print step,action
        # print "dP"," ".join("%.6f"%norm((pn-p)/stepper.step) for pn,p in zip(P,Pnext))
        # print "dE"," ".join("%.1f"%(en-e) for en,e in zip(E,Enext))
        # print "En"," ".join("%.1f"%e for e in Enext)
        # print "accept",accept
        #print("T0 %12.2f %12.2f ... Tmax %12.2f %12.2f"%(Enext[0], (Enext[0] - E[0])/T[0] if Enext[0]>E[0] else 0,
        #                                                 Enext[-1], (Enext[-1] - E[-1])/T[-1] if Enext[-1] > E[-1] else 0.),
        #      action)
        acceptance_frequency += accept
        E[accept] = Enext[accept]
        P[accept] = Pnext[accept]
        if step > burn :
            energy_change += Enext - E
            step_size += np.abs(np.average(asarray(delta)/scale, axis=1))
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
            swap = zeros(N)
            #for i in range(N - 2, -1, -1):
            for i in np.random.permutation(N-1):
                #if not swap[i+1] and exp((E[i + 1] - E[i])* dT[i]) > rand():
                #if rand()>0.5 and exp((E[i + 1] - E[i])* dT[i]) > rand():
                above = (i+1)%N
                if exp((E[above] - E[i])* dT[i]) > rand():
                    swap[i] = 1
                    # switch the energy states around
                    swap_history[above], swap_history[i] = swap_history[i], swap_history[above]
                    labels[above], labels[i] = labels[i], labels[above]

                    E[above], E[i] = E[i], E[above]
                    P[above], P[i] = P[i] + 0, P[above] + 0
                    swap_frequency[above] += 1
                    swap_frequency[i] += 1
            total_swap += swap[:-1]
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
        self.temperatures, self.points, self.energies, self.swap, self.unnorm_points, self.unnorm_energy = [], [], [], [], [], []
        self.var_labels = var_labels
        self.file = open("scale.txt", "w")
        self.swap_hist = []
        self.staggered_temp = []
        self.slope = 0
        self.intercept = 0
    def swap_save(self, temp, value, slope, intercept):
        self.swap_hist.extend(value)
        self.staggered_temp.extend(temp)
        self.intercept = intercept
        self.slope = slope

    def save(self, step, temperature, energy, point, swap_history, labels, changed=None):
        if step > self.burn:
            swap = swap_history.argsort()
            self.temperatures.append(temperature.tolist())
            self.unnorm_points.append(point.tolist())
            self.unnorm_energy.append(energy.tolist())

            self.points.append(point[swap])
            self.energies.append(energy[swap])

            self.swap.append(swap)

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
        from pylab import plot, figure, semilogy, ylabel, xlabel, suptitle, gca, hist, subplot, legend, cm, quiver, yscale
        from dream import  views
        from numpy import array
        from scipy import var
        from statsmodels.robust.scale import mad

        # Parameter History
        figure()
        gca().set_color_cycle([cm.gist_ncar(i) for i in np.linspace(0, 0.9, len(self.points[0][0]))])
        generation = np.linspace(0, len(self.points) - 1, len(self.points))
        graphs = []
        for i in range(len(self.points[0])):
            for j in range(len(self.points[0][0])-1, -1, -1):
                graph, = plot(generation, map(lambda p:p[i][j], self.points), hold=True, marker='.', markersize=1)
                graphs.append(graph)
        group = [tuple(graphs[::i+1]) for i in range(len(self.points[0][0]))]
        legend(group, self.var_labels, markerscale=10)
        ylabel("Value")
        xlabel("Generation")
        suptitle("Parameter History")

        # Energy History
        figure()
        for i in range(len(self.energies[0])):
            plot(generation, map(lambda e:e[i]*-1, self.energies), ',', hold=True)
        ylabel("Value")
        xlabel("Generation")
        suptitle("Log Likelihood")

        # Swap History
        figure()
        for i in range(len(self.swap[0])):
            semilogy(generation, map(lambda s,t:t[s[i]], self.swap, self.temperatures), hold=True)
        suptitle("Temperature Swap History")
        ylabel("Temperature")
        xlabel("Generation")

        def directed_plot(x,y):
            x,y = x[-1000:], y[-1000:]
            quiver(x[:-1],y[:-1],diff(x),diff(y), scale_units='xy', angles='xy', scale=1, width=0.001)
        figure()
        points = array(self.points)
        for i in range(1):
            print(points.shape)
            # semilogy(points[:,i,0], map(lambda s,t : t[s[i]], self.swap, self.temperatures) )
            directed_plot(points[:,i,0], map(lambda s,t : t[s[i]], self.swap, self.temperatures))
        suptitle("Temperature Swap History")
        yscale('log')
        ylabel("Temperature")
        xlabel("Generation")


        nw, nh = views.tile_axes(len(self.unnorm_points[0][0]))
        figure()
        for i in range(len(self.unnorm_points[0][0])):
            subplot(nw, nh, i+1)
            parameter = []
            map(lambda p:parameter.extend([t[i] for t in p]), self.unnorm_points)
            semilogy(parameter, asarray(self.temperatures).flatten(), '.', hold=True)
        xlabel("Value")
        ylabel("Temperature")

        rot = asarray(self.temperatures).flatten().argsort()

        parameter = array(parameter)[rot]
        parameters = np.split(parameter, len(self.temperatures[1]))

        variance = [var(parameters[i]) for i in range(len(parameters))]
        print(self.temperatures[1])
        print(array(self.temperatures[1]))
        z = np.polyfit((self.temperatures[1]), array(variance), 3)
        p = np.poly1d(z)
        x2 = np.linspace(self.temperatures[1][0], self.temperatures[1][-1], 1000)
        y = p(x2)
        figure()

        semilogx(self.temperatures[1], variance, '.', hold=True)
        semilogx(x2, y)

        # # 1D histograms
        # figure()
        # nw, nh = views.tile_axes(len(self.unnorm_points[0][0]))
        # weights = None
        # cbar = views._make_fig_colorbar(asarray(self.unnorm_energy).flatten() * -1)
        # ONE_SIGMA = 1 - 2*0.15865525393145705
        # parameters = []
        #
        # for i in range(len(self.unnorm_points[0][0])):
        #     parameter = []
        #     map(lambda p:parameter.extend([t[i] for t in p]), self.unnorm_points)
        #     parameters.append(parameter)
        #
        #     subplot(nw, nh, i+1)
        #     p100, p68, p0 = stats.credible_intervals(x=asarray(parameter), weights=weights, ci=[0.9999, ONE_SIGMA, 0.0])
        #     mean, std = stats.stats(x=asarray(parameter), weights=weights)
        #
        #     views._make_logp_histogram(asarray(parameter), asarray(self.unnorm_energy).flatten() * -1, 80, p100, weights, cbar)
        #
        #     views._decorate_histogram(stats.VarStats(label=self.var_labels[i], index=i+1, p95=p100, p68=p68,
        #                               median=p0[0], mean=mean, std=std, best=self.best_point[i]))

        #
        # # 2D histograms
        # figure()
        # corrplot.Corr2d(parameters, bins=50, labels=self.var_labels).plot()


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
        return delta

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
        return delta

    def jiggle(self, p, noise):
        delta = randn(len(p)) * self.step * noise
        assert norm(delta) != 0
        return delta

    def random(self, p):
        delta = rand(len(p)) * self.step + self.offset
        assert norm(delta) != 0
        return delta

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
        return delta


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
