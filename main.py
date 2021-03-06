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


def plotSRCNComputedHalfspace(generator_halfspace, noise_halfspace, computed_halfspace, data, labels):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    x = np.linspace(-10, 10, 100)
    gen_y = x * generator_halfspace
    noise_y = x * noise_halfspace
    comp_y = x * computed_halfspace
    for i in range(len(data)):
        if labels[i] == 1:
            plt.plot(data[i][0], data[i][1], 'g+', mew=1, ms=10)
        elif labels[i] == -1:
            plt.plot(data[i][0], data[i][1], 'b_', mew=1, ms=10)
        else:
            print('something went wrong')
    plt.plot(x, gen_y, 'r--', label='label')
    plt.plot(x, comp_y, 'r', label='comp')
    plt.plot(x, noise_y, 'k--', label='noise')
    plt.legend(loc='lower right')
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    # plt.tight_layout(rect=[0, 0, 1, 1])
    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()


# add RCN noise by flipping labels w.p. eta
def addRCN(labels, eta):
    noisy_labels = labels.copy()
    for idx in range(len(noisy_labels)):
        if np.random.uniform(0, 1) < eta:
            noisy_labels[idx] *= -1
    return noisy_labels


# add SRCN halfspace noise w.p. eta.
# param: halfspace is slope of some labeling generator hyperplane passing through origin
def addSRCNHalfspace(data, labels, eta, halfspace):
    noisy_labels = labels.copy()
    # find vec orthogonal to halfspace slope through origin
    normal = np.array([1, halfspace])
    dots = np.array([np.dot(normal, data[i]) for i in range(len(data))])
    applyNoise = np.sign(dots)

    for idx in range(len(applyNoise)):
        if applyNoise[idx] == 1 and np.random.uniform(0, 1) < eta:
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
    dim = len(data[0])
    x0 = np.array([0.1 for d in range(dim)])
    # cons = (
    #     {'type': 'ineq',
    #      'fun': lambda x: np.linalg.norm(x, ord=2) - 1}
    # )
    res = minimize(rcnObj,
                   x0,
                   args=tuple([lam, data, labels]),
                   options={'disp': True},
                   tol=1e-8)

    return res.x


def accuracy(data, labels, w):
    preds = np.array([np.sign(np.dot(w, data[idx])) for idx in range(len(data))])
    return np.count_nonzero(preds == labels) / len(data)


# generate synthetic data iter times and return average of all accuracies
def experiment2dimSynthetic(iter, eta, plot=False):
    gen_og = []
    gen_noisy = []
    comp_og = []
    comp_noisy = []
    for _ in range(iter):
        # generate random halfspace through origin by specifying a slope
        theta = np.random.uniform(0, 360)
        slope = (np.sin(theta) / np.cos(theta))
        normal_w = np.array([1, -1/slope])
        # use halfspace to label data
        lam = eta
        halfspace, data, labels = makeLinearData(75, slope)
        # noisy_labels = addRCN(labels, eta)

        # theta_noise = np.random.uniform(0, 360)
        # slope_noise = (np.sin(theta_noise) / np.cos(theta_noise))
        noisy_labels = addSRCNHalfspace(data, labels, eta, halfspace)
        comp_normal_w = rcnOptimize(lam, data, labels)

        gen_og.append(accuracy(data, labels, normal_w))
        gen_noisy.append(accuracy(data, noisy_labels, normal_w))
        comp_og.append(accuracy(data, labels, comp_normal_w))
        comp_noisy.append(accuracy(data, noisy_labels, comp_normal_w))
        print('acc of gen on og', gen_og[-1])
        print('acc of gen on noisy:', gen_noisy[-1])
        print('acc of comp on og', comp_og[-1])
        print('acc of comp on noisy:', comp_noisy[-1])

        if plot:
            w_slope = comp_normal_w[1] / comp_normal_w[0]
            print(slope, -1/w_slope)
            plotSRCNComputedHalfspace(slope, -1/slope, -1/w_slope, data, noisy_labels)
            # plotComputedHalfspace(slope, -1/w_slope, data, noisy_labels)

    print('====')
    print('eta =', eta)
    print('avg acc gen on og', sum(gen_og) / len(gen_og))
    print('avg acc gen on noisy', sum(gen_noisy) / len(gen_noisy))
    print('avg acc comp on og', sum(comp_og) / len(comp_og))
    print('avg acc comp on noisy', sum(comp_noisy) / len(comp_noisy))


if __name__ == '__main__':
    experiment2dimSynthetic(1, 0.3, True)
