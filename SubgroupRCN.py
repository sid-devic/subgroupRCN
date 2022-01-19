import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# uniformly sample num_points from nd cube and label with halfspace given by normal
def uniformLinearData(dim, num_points, normal):
    if dim != len(normal):
        print('supply a normal with dimension', dim)
        print('===')
        return

    data = np.random.uniform(-1, 1, (num_points, dim))
    dots = np.array([np.dot(normal, data[i]) for i in range(len(data))])
    labels = np.sign(dots)

    return data, labels


def plot2dLinearData(normal, data, labels):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    x = np.linspace(-10, 10, 100)
    y = x * -(normal[0] / normal[1])

    for i in range(len(data)):
        if labels[i] == 1:
            plt.plot(data[i][0], data[i][1], 'g+', mew=1, ms=10)
        elif labels[i] == -1:
            plt.plot(data[i][0], data[i][1], 'b_', mew=1, ms=10)
        else:
            print('something went wrong')

    # plot halfspace
    plt.plot(x, y, 'r')
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.show()


def plotComputedHalfspace(generator_normal, computed_normal, data, labels):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    x = np.linspace(-10, 10, 100)
    gen_y = x * -(generator_normal[0] / generator_normal[1])
    comp_y = x * -(computed_normal[0] / computed_normal[1])
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


def plotSRCNComputedHalfspace(generator_normal, noise_normal, computed_normal, data, labels):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    x = np.linspace(-10, 10, 100)
    gen_y = x * -(generator_normal[0] / generator_normal[1])
    noise_y = x * -(noise_normal[0] / noise_normal[1])
    comp_y = x * -(computed_normal[0] / computed_normal[1])
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
# param: halfspace is normal of some labeling generator hyperplane passing through origin
def addSRCNHalfspace(data, labels, eta, normal):
    noisy_labels = labels.copy()
    # find vec orthogonal to halfspace slope through origin
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
def rcnOptimize(lam, data, labels, verbose):
    dim = len(data[0])
    x0 = np.array([0.1 for d in range(dim)])
    # cons = (
    #     {'type': 'ineq',
    #      'fun': lambda x: np.linalg.norm(x, ord=2) - 1}
    # )
    res = minimize(rcnObj,
                   x0,
                   args=tuple([lam, data, labels]),
                   options={'disp': verbose}
                   )

    return res.x


def accuracy(data, labels, w):
    preds = np.array([np.sign(np.dot(w, data[idx])) for idx in range(len(data))])
    return np.count_nonzero(preds == labels) / len(data)


# generate synthetic data iter times and return average of all accuracies
def runExperiment(dim, iter, points_per_iter, eta, noiseType, plot=False, verbose=False):
    gen_og = []
    gen_noisy = []
    comp_og = []
    comp_noisy = []
    orthogonalNoise = False

    for _ in range(iter):
        # generate random normal for labeling points
        normal_w = np.random.uniform(-1, 1, (dim, ))
        # use halfspace to label data
        # convergence guarantees for leakyRelu of param lam=eta see Appx. A: https://arxiv.org/pdf/1906.10075.pdf
        lam = eta
        data, labels = uniformLinearData(dim=dim, num_points=points_per_iter, normal=normal_w)

        if noiseType == 'RCN':
            noisy_labels = addRCN(labels, eta)
        elif noiseType == 'SRCN':
            if orthogonalNoise:
                noise_normal_w = 0
            else:
                # random normal for noise
                noise_normal_w = np.random.uniform(-1, 1, (dim,))
            noisy_labels = addSRCNHalfspace(data, labels, eta, noise_normal_w)
        else:
            print('unexpected noise type ', noiseType)
            raise

        comp_normal_w = rcnOptimize(lam, data, labels, verbose)
        gen_og.append(accuracy(data, labels, normal_w))
        gen_noisy.append(accuracy(data, noisy_labels, normal_w))
        comp_og.append(accuracy(data, labels, comp_normal_w))
        comp_noisy.append(accuracy(data, noisy_labels, comp_normal_w))
        if verbose:
            print('normal', normal_w)
            print('acc of gen on og', gen_og[-1])
            print('acc of gen on noisy:', gen_noisy[-1])
            print('acc of comp on og', comp_og[-1])
            print('acc of comp on noisy:', comp_noisy[-1])

        if plot and dim == 2:
            if noiseType == 'RCN':
                plotComputedHalfspace(normal_w, comp_normal_w, data, noisy_labels)
            elif noiseType == 'SRCN':
                plotSRCNComputedHalfspace(normal_w, noise_normal_w, comp_normal_w, data, noisy_labels)
            else:
                print('unexpected noise type ', noiseType)
                raise

    table_data = [
        ['dim={0}'.format(dim), 'noise={0} eta={1}'.format(noiseType, eta), 'iter={0}'.format(iter)],
        ['==============================', '', ''],
        ['halfspace', 'noise', 'acc'],
        ['gen', 'og', round(sum(gen_og) / len(gen_og), 3)],
        ['gen', eta, round(sum(gen_noisy) / len(gen_noisy), 3)],
        ['comp', 'og', round(sum(comp_og) / len(comp_og), 3)],
        ['comp', eta, round(sum(comp_noisy) / len(comp_noisy), 3)]
    ]
    print('==============================')
    for row in table_data:
        print("{: >8} {: >8} {: >8}".format(*row))


if __name__ == '__main__':
    for eta in [0, 0.1, 0.2, 0.3, 0.4, 0.45]:
        runExperiment(dim=10,
                      iter=100,
                      points_per_iter=75,
                      eta=eta,
                      noiseType='SRCN',
                      plot=False)
