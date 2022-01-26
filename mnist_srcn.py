from mnist import MNIST
import matplotlib.pyplot as plt
import numpy as np

from synthetic_srcn import accuracy, addRCN, addSRCNHalfspace, rcnOptimize

# helper functions from:
# https://www.programcreek.com/python/?code=eliben%2Fdeep-learning-samples%2Fdeep-learning-samples-master%2Flogistic-regression%2Fmnist_dataset.py
def display_mnist_image(x, y=None):
    """Displays a single mnist image with a label.

    x: (784,) image vector, as stored in the mnist pickle.
    y: optional numeric label
    """
    xmat = x.reshape(28, 28)
    plt.imshow(xmat, cmap='gray')
    if y is not None:
        plt.title('label={0}'.format(y))
    plt.show()


def display_multiple_images(xs):
    """Displays multiple images side-by-side in subplots."""
    fig = plt.figure()
    fig.set_tight_layout(True)

    for i, x in enumerate(xs):
        ax = fig.add_subplot(1, len(xs), i + 1)
        ax.imshow(x.reshape(28, 28), cmap='gray')
    plt.show()


def runExperiment(data, labels, iter, eta, noiseType, verbose=False):
    comp_og = []
    comp_noisy = []
    dim = data.shape[1]

    for _ in range(iter):
        # convergence guarantees for leakyRelu of param lam=eta see Appx. A: https://arxiv.org/pdf/1906.10075.pdf
        lam = eta

        if noiseType == 'RCN':
            print('adding RCN noise eta={0}'.format(eta))
            noisy_labels = addRCN(labels, eta)
        elif noiseType == 'SRCN':
            # random normal for noise
            noise_normal_w = np.random.uniform(-1, 1, (dim,))
            noisy_labels = addSRCNHalfspace(data, labels, eta, noise_normal_w)
            disagreements = np.count_nonzero(noisy_labels == labels)
            print('num disagreements on noisy labels:', disagreements)
        else:
            print('unexpected noise type ', noiseType)
            raise

        comp_normal_w = rcnOptimize(lam, data, labels, verbose)
        comp_og.append(accuracy(data, labels, comp_normal_w))
        comp_noisy.append(accuracy(data, noisy_labels, comp_normal_w))
        if verbose:
            print('acc of comp on og', comp_og[-1])
            print('acc of comp on noisy:', comp_noisy[-1])

    table_data = [
        ['', 'noise={0} eta={1}'.format(noiseType, eta), 'iter={0}'.format(iter)],
        ['==============================', '', ''],
        ['halfspace', 'noise', 'acc'],
        ['comp', 'og', round(sum(comp_og) / len(comp_og), 3)],
        ['comp', eta, round(sum(comp_noisy) / len(comp_noisy), 3)]
    ]
    print('==============================')
    for row in table_data:
        print("{: >8} {: >8} {: >8}".format(*row))


def mnist():
    # make sure to download all four files from: http://yann.lecun.com/exdb/mnist/
    # (train data, train labels, test data, test labels) and save them in data/ folder.
    # make sure you also unzip all the data files (can just use `gunzip *`).
    mndata = MNIST('data/')
    images, labels = mndata.load_training()
    images = np.array(images)
    labels = np.array(labels)

    # display_mnist_image(images[0], labels[0])
    # get only zeros and ones from dataset
    idx_zeros = np.where(labels == 0)
    idx_ones = np.where(labels == 1)
    idxs = np.append(idx_zeros, idx_ones)
    images = images[idxs]
    labels = labels[idxs].astype(float)

    # how many points to use in (randomly selected) train set
    num_points = 500
    random_perm = np.random.permutation(len(idxs))
    x, y = images[random_perm], labels[random_perm]
    x = x[:num_points]
    y = y[:num_points]

    # change zero labels to -1 for nice binary classification
    y[np.where(y == 0)] = -1
    print(y)

    runExperiment(data=x,
                  labels=y,
                  iter=1,
                  eta=0.2,
                  noiseType='SRCN',
                  verbose=True)


if __name__ == '__main__':
    mnist()
