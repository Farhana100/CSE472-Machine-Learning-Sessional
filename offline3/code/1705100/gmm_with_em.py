import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA

DATASET_FILENAME = 'data2D.csv'


# DATASET_FILENAME = 'data3D.csv'
# DATASET_FILENAME = 'data6D.csv'

# ---------------------------------------- E-step: Expectation Calculation start ----------------------------------->>>

def compute_probabilities(X, priors, means, covars):
    """
    computes the probabilities of samples belonging to a certain cluster for the current iteration.
    The output probabilities are normalized for every sample (x)
    i.e., sum(p_ij) = 1

    :param X: Data matrix. (n x m) = (#ofsamples x #offeatures)
    :param priors: weights (proportion of samples belonging to a cluster) of each cluster for current iteration.
                (1 x k) = (1 x #ofcluster)
    :param means: Means of each cluster for current iteration. (k x m) = (#ofcluster x #offeatures)
    :param covars: Covariances of each cluster for current iteration. (k x (m x m)) = (#ofcluster x (m x m))
    :return: probabilities (normalized). (n x k)
    """

    X = np.array(X)
    priors = np.array(priors).ravel()
    means = np.array(means)
    covars = np.array(covars)

    n, m = X.shape
    k = len(means)

    # initialize probabilities with zeros
    prob = np.zeros((n, k))

    # prob = np.array([priors[j] * multivariate_normal.pdf(x, means[j], covars[j]) for j in range(k) for x in X])

    for i in range(n):
        for j in range(k):
            prob[i, j] = priors[j] * multivariate_normal.pdf(X[i], means[j], covars[j])

    # normalize
    row_sums = prob.sum(axis=1)[:, np.newaxis]
    prob = prob / row_sums

    return prob


# ---------------------------------------- E-step: Expectation Calculation end ------------------------------------->>>


# ------------------------------------------ M-step: Maximization start ------------------------------------------->>>

def compute_priors(prob):
    """
    compute the new proportion of points belonging to each cluster for this iteration.
    :param prob: probabilities (normalized). (n x k) = (#ofsamples x #ofclusters)
    :return: new priors  (1 x k) = (1 x #ofclusters)
    """
    prob = np.array(prob)
    n, k = prob.shape
    new_priors = np.sum(prob, axis=0) / n

    return new_priors


def compute_means(X, prob):
    """
    compute the updated means of each cluster for this iteration.
    The new mean of each cluster is the weighted average of all data points, weighted by the cluster probabilities.

    :param X: Data matrix. (n x m) = (#ofsamples x #offeatures)
    :param prob: probabilities (normalized). (n x k) = (#ofsamples x #ofclusters)
    :return: new means  (k x m) = (#ofcluster x #offeatures)
    """

    X = np.array(X)
    prob = np.array(prob)

    new_means = np.matmul(prob.T, X) / np.sum(prob, axis=0)[:, np.newaxis]

    return new_means


def compute_covariances(X, prob, means):
    """
    computing the new covariances
    :param X: Data matrix. (n x m) = (#ofsamples x #offeatures)
    :param prob: probabilities (normalized). (n x k) = (#ofsamples x #ofclusters)
    :param means: means  (k x m) = (#ofcluster x #offeatures)
    :return: covars: Covariances of each cluster for current iteration. (k x (m x m)) = (#ofcluster x (m x m))
    """

    X = np.array(X)
    prob = np.array(prob)
    means = np.array(means)

    n, m = X.shape
    k, _ = means.shape

    new_covariances = []
    prob_sum = prob.sum(axis=0)

    for j in range(k):
        weighted_sum = np.zeros((m, m), float)

        for i in range(n):
            deviation = (X[i] - means[j])[:, np.newaxis]
            weighted_sum += np.matmul(deviation, deviation.T) * prob[i][j]

        new_covariances.append(weighted_sum / prob_sum[j])

    return np.array(new_covariances)


# ------------------------------------------ M-step: Maximization end --------------------------------------------->>>


# ------------------------------------------ Cost Function start --------------------------------------------->>>

def calc_loglikelihood(X, priors, means, covars):
    """
    Compute loglikelihood
    :param X: Data matrix. (n x m) = (#ofsamples x #offeatures)
    :param priors: weights (proportion of samples belonging to a cluster) of each cluster for current iteration.
                (1 x k) = (1 x #ofcluster)
    :param means: means  (k x m) = (#ofcluster x #offeatures)
    :param covars: Covariances of each cluster for current iteration. (k x (m x m)) = (#ofcluster x (m x m))
    :return:
    """

    X = np.array(X)
    priors = np.array(priors).ravel()
    means = np.array(means)
    covars = np.array(covars)

    n, m = X.shape
    k, _ = means.shape

    ll = 0

    for i in range(n):
        temp = []
        for j in range(k):
            deviation = np.array(X[i] - means[j])
            exp = np.dot(deviation.T, np.dot(np.linalg.inv(covars[j]), deviation))

            temp.append(np.log(priors[j]) - 0.5 * (exp + m * np.log(2 * np.pi) + np.log(np.linalg.det(covars[j]))))

        temp = np.array(temp)
        ll += np.max(temp) + np.log(np.sum(np.exp(temp - np.max(temp))))

    return ll


# ------------------------------------------ Cost Function end ----------------------------------------------->>>


# ------------------------------------------ EM algorithm start ----------------------------------------------->>>

def EM_algo(X, k, maxiter=1000, thresh=1e-4):
    """
    Fit GMM
    :param X:  Data matrix. (n x m) = (#ofsamples x #offeatures)
    :param k:  #ofClusters
    :param maxiter: maximum iteration
    :param thresh:
    :return:
    """

    X = np.array(X)
    n, m = X.shape

    # initialization
    # means = np.zeros((k, m))
    np.random.seed(4)

    # Initialization of parameters
    chosen = np.random.choice(len(X), k, replace=False)
    means = [X[i] for i in chosen]
    covars = np.array([np.eye(m)] * k)
    priors = np.ones((1, k)) / k
    prob = np.full((n, k), 1 / k)

    ll = calc_loglikelihood(X, priors, means, covars)
    allll = [ll]

    for i in range(maxiter):
        if i % 10 == 0:
            print('at iteration', i)

        # E-step: Expectation Calculation
        prob = compute_probabilities(X, priors, means, covars)

        # M-step: Maximization

        # Update priors, means, covariances
        priors = compute_priors(prob)
        means = compute_means(X, prob)
        covars = compute_covariances(X, prob, means)

        # loglikelihood
        ll = calc_loglikelihood(X, priors, means, covars)

        # Check for convergence in log-likelihood and store
        if (ll - allll[-1]) < thresh and ll > -np.inf:
            break
        allll.append(ll)

    out = {'priors': priors, 'means': means, 'covariances': covars, 'llike': allll, 'probabilities': prob}

    return out


# -------------------------------------------- EM algorithm end ----------------------------------------------->>>

# -------------------------------------------- plotting start ----------------------------------------------->>>

def normal_dist(X, Y, mean, covar):
    var = multivariate_normal(mean=mean, cov=covar)

    Z = np.array([[var.pdf([X[i][j], Y[i][j]]) for j in range(X.shape[1])] for i in range(X.shape[0])])
    return Z


def plot_contours(X, means, covars, figId, title=None, save=False):
    X = np.array(X)
    means = np.array(means)
    covars = np.array(covars)

    plt.figure(figId)

    plt.plot(X[:, 0], X[:, 1], ',')

    delta = (max(np.max(X[:, 0]), np.max(X[:, 1])) - min(np.min(X[:, 0]), np.min(X[:, 1]))) / 100
    k, _ = means.shape

    x = np.arange(np.min(X[:, 0]), np.max(X[:, 0]), delta)
    y = np.arange(np.min(X[:, 1]), np.max(X[:, 1]), delta)
    X, Y = np.meshgrid(x, y)

    for i in range(k):
        Z = normal_dist(X, Y, means[i], covars[i])
        plt.contour(X, Y, Z)

    if title:
        plt.title(title)
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()

    if save:
        plt.savefig(title + '.png')


# -------------------------------------------- plotting end ----------------------------------------------->>>


def load_dataset(filename):
    """
    function for reading data from csv
    :param filename: dataset filename   n x m
    :return: X
    """
    X = []
    with open(filename, 'r') as f:
        for line in f:
            X.append(line.strip().split(' '))

    X = np.array([[float(j) for j in i] for i in X])

    return X


if __name__ == '__main__':
    X = load_dataset(DATASET_FILENAME)

    # print(X.shape)
    # print(X)
    plt.figure(1)
    plt.plot(X[:, 0], X[:, 1], 'ko')
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()
    plt.savefig('test.png')
    # plt.show()

    max_k = int(input('enter k: '))

    convergedloglikelihood = []

    for k in range(2, max_k+1):
        print('k =', k, 'running EM')
        out = EM_algo(X, k, maxiter=100)
        print(out['means'])

        ll = out['llike']
        convergedloglikelihood.append(ll[-1])

    # # plotting
    # plot_contours(X, means=out['means'], covars=out['covariances'], figId=2, title='contours', save=True)
    # # print(out)

    plt.figure(3)
    plt.plot(range(2, max_k+1), convergedloglikelihood, linewidth=4)
    plt.xlabel('number of clusters')
    plt.ylabel('Log-likelihood')
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()

    plt.savefig('Converged loglikelihood vs k.png')
