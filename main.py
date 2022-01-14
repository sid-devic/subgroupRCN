import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# make 2d linearly separable dataset inside square of length 2 centered at origin
def makeLinearData(num_points, slope):
    # find vec orthogonal to halfspace slope through origin
    normal = np.array([1, -1/slope])
    data = np.random.uniform(-1, 1, (num_points, 2))
    dots = np.array([np.dot(normal, data[i]) for i in range(len(data))])
    labels = np.sign(dots)

    return slope, data, labels


def plotLinearData(halfspace, data, labels):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    x = np.linspace(-10, 10, 100)
    y = x * halfspace
    for i in range(len(data)):
        if labels[i] == 1:
            plt.plot(data[i][0], data[i][1], 'g+', mew=1, ms=10)
        elif labels[i] == -1:
            plt.plot(data[i][0], data[i][1], 'b_', mew=1, ms=10)
        else:
            print('something went wrong')
    plt.plot(x, y, 'r')
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.show()


def plotComputedHalfspace(generator_halfspace, computed_halfspace, data, labels):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    x = np.linspace(-10, 10, 100)
    gen_y = x * generator_halfspace
    comp_y = x * computed_halfspace
    for i in range(len(data)):
        if labels[i] == 1:
            plt.plot(data[i][0], data[i][1], 'g+', mew=1, ms=10)
        elif labels[i] == -1:
            plt.plot(data[i][0], data[i][1], 'b_', mew=1, ms=10)
        else:
            print('something went wrong')
    plt.plot(x, gen_y, 'r--', label='generating halfspace')
    plt.plot(x, comp_y, 'r', label='computed halfspace')
    plt.legend()
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.show()


# add RCN noise by flipping labels w.p. eta
def addRCN(labels, eta):
    noisy_labels = labels.copy()
    for idx in range(len(noisy_labels)):
        if np.random.uniform(0, 1) < eta:
            noisy_labels[idx] *= -1
    return noisy_labels


def rcnObj(w, lam, data, labels):
    # define leaky relu function
    def leakyRelu(x):
        if x >= 0:
            return (1 - lam) * x
        else:
            return lam * x

    n = len(data)
    preds = np.array([leakyRelu(-1 * labels[idx] * np.dot(w, data[idx])) for idx in range(n)])
    return (1 / n) * np.sum(preds)


# optimize leaky relu with sklearn.optimize to find halfspace on noisy data
def rcnOptimize(lam, data, labels):
    x0 = np.array([0.1, 0.1])
    cons = (
        {'type': 'ineq',
         'fun': lambda x: np.linalg.norm(x, ord=2) - 1}
    )
    res = minimize(rcnObj,
                   x0,
                   args=tuple([lam, data, labels]),
                   options={'disp': True},
                   tol=1e-8)

    return res.x


if __name__ == '__main__':
    # generate random halfspace through origin by specifying a slope
    theta = np.random.uniform(0, 360)
    slope = (np.sin(theta) / np.cos(theta))
    # use halfspace to label data
    eta = 0.45
    lam = eta
    halfspace, data, labels = makeLinearData(50, slope)
    noisy_labels = addRCN(labels, eta)

    normal_w = rcnOptimize(lam, data, labels)
    w_slope = normal_w[1] / normal_w[0]
    print(slope, -1/w_slope)
    plotComputedHalfspace(slope, -1/w_slope, data, noisy_labels)
