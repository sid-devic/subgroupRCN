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


# synthetic experiment input distribution from ankur's paper
# https://github.com/secanth/massart/blob/master/experiment.py
# d: dimension of instance
# N: number of points in train set
# frac: 0.25*N is number of points in test set
def mixture_gauss(d, N, frac=0.25):
    total = int(N * (frac + 1))
    cov1 = np.eye(d)
    cov2 = np.eye(d)
    cov2[0, 0] = 8.
    cov2[0, 1] = 0.1
    cov2[1, 0] = 0.1
    cov2[1, 1] = 0.0024
    vecs = np.zeros((total, d))
    for i in range(total):
        if np.random.uniform() > 0.5:
            vecs[i, :] = np.random.multivariate_normal([0] * d, cov1)
        else:
            vecs[i, :] = np.random.multivariate_normal([0] * d, cov2)
    x_train = vecs[:N, :]
    x_test = vecs[N:, :]
    y_train = (vecs[:N, 1] > 0).astype(int) * 2 - 1
    y_test = (vecs[N:, 1] > 0).astype(int) * 2 - 1

    return x_train, x_test, y_train, y_test


def plot2dLinearData(normal, data, labels, lims=None):
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
    if lims is not None:
        plt.xlim([-lims[0], lims[0]])
        plt.ylim([-lims[1], lims[1]])
    else:
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
        normal_w = np.random.uniform(-1, 1, (dim,))
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


def runMassartExperiment(dim, iter, points_per_iter, eta, plot=False):
    train_acc = []
    noisy_train_acc = []
    noisy_test_acc = []

    for _ in range(iter):
        x_train, x_test, y_train, y_test = mixture_gauss(dim, points_per_iter)
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        lam = eta

        def massartCorrupt(xs, ys):
            noisy_y = ys.copy()
            # apply massart noise as in ankur's paper
            for idx in range(len(ys)):
                if xs[idx, 1] > 0.3 and np.random.uniform(0, 1) < eta:
                    noisy_y[idx] *= -1
            return noisy_y

        noisy_y_train = massartCorrupt(x_train, y_train)
        noisy_y_test = massartCorrupt(x_test, y_test)

        comp_normal_w = rcnOptimize(lam, x_train, noisy_y_train, verbose=False)
        train_acc.append(accuracy(x_train, y_train, comp_normal_w))
        noisy_train_acc.append(accuracy(x_train, noisy_y_train, comp_normal_w))
        noisy_test_acc.append(accuracy(x_test, noisy_y_test, comp_normal_w))

        if plot:
            x_lim = np.abs(x_train[:, 0]).max()
            y_lim = np.abs(x_train[:, 1]).max()
            plot2dLinearData(comp_normal_w, x_train, noisy_y_train, lims=[x_lim, y_lim])

    print('avg accuracy on no noise train', sum(train_acc) / len(train_acc))
    print('avg accuracy on noisy train eta={0}'.format(eta), sum(noisy_train_acc) / len(noisy_train_acc))
    print('avg accuracy on noisy test eta={0}'.format(eta), sum(noisy_test_acc) / len(noisy_test_acc))
    print('')

if __name__ == '__main__':
    # for eta in [0, 0.1, 0.2, 0.3, 0.4, 0.45]:
    #     runExperiment(dim=10,
    #                   iter=100,
    #                   points_per_iter=75,
    #                   eta=eta,
    #                   noiseType='RCN',
    #                   plot=False)
    for eta in [0.05 * i for i in range(10)]:
        runMassartExperiment(dim=2, iter=1, points_per_iter=1000, eta=eta, plot=True)
