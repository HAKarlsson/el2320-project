import numpy as np
from numpy.linalg import norm
from numpy.random import uniform, normal, randint
from sklearn.cluster import KMeans
from scipy.cluster.vq import kmeans2
from aux_func import normImg
from math import ceil


def initS(M, xmin, xmax, ymin, ymax, d=3):
    """
    Initialize particles uniformaly distributed in (xmin, xmax, ymin, ymax)
    """
    S = np.zeros((M, d))
    S[:, 0] = uniform(xmin, xmax, M)
    S[:, 1] = uniform(ymin, ymax, M)
    S[:, 2] = 1. / M
    return S


def boundS(S, xmin, xmax, ymin, ymax):
    """
    Bound particles S to (xmin, xmax, ymin, ymax)
    """
    S[S[:, 0] < xmin, 0] = xmin
    S[S[:, 1] < ymin, 1] = ymin
    S[S[:, 0] >= xmax, 0] = xmax - 1e-9
    S[S[:, 1] >= ymax, 1] = ymax - 1e-9


def sys_resample(S, M, w=2):
    """
    Systematic resampling of S. 
    M is the number of particles to resample
    w is the location of the weights

    """
    cdf = create_cdf(S[:, w])
    Sp = np.zeros((M, S.shape[1]))
    i, r = 0, uniform()
    for m in range(M):
        while (m + r) / M > cdf[i]:
            i += 1
        Sp[m, :] = S[i, :]
    Sp[:, 2] = 1. / M
    return Sp


def create_cdf(weights):
    """
    Create a cumulative density function our of the weights
    """
    cdf = weights / np.sum(weights)
    for n in range(1, weights.shape[0]):
        cdf[n] += cdf[n-1]
    return cdf


def clusterWeight(weights, labels, C):
    """
    Calculate the weight of each cluster.
    """
    c_weight = np.zeros(C)
    for c in range(C):
        c_weight[c] = np.sum(weights[labels == c])
    return c_weight / np.sum(c_weight)



def distribute(weights, N):
    # distribute the samples exactly
    M = weights.shape[0]
    distri = np.zeros(M, dtype=int)
    rollover = 0.
    for c in range(M):
        size = weights[c] * N + rollover
        distri[c] = np.round(size)
        rollover = size - distri[c]
    return distri


def clusterScore(clusters, label, samples):
    """
    average square distance of particle to their assigned cluster
    """
    return np.mean(norm(samples - clusters[label, :], axis=1)**2)

# Offset for selecting pixels
OFFSETS = (720 * np.repeat(np.arange(32), 32) +
           np.tile(np.arange(32), 32)).reshape(1, -1)


def frame2weights(predictor, S, frame):
    """
    Predict the probability of an ant for each particle S given the frame and predictor
    """
    M = S.shape[0]
    frame = normImg(frame).flatten()
    ind = S[:, 0:2].astype(int)
    ind -= 16
    xs, ys = ind[:, 1], ind[:, 0]
    ind = (xs + ys * 720).reshape(-1, 1)
    ind = ind + OFFSETS
    # should vectorize loop
    return predictor.predict_proba(frame[ind])[:, 1]


class ATBase:
    """
    Base class for ant tracking
    """

    def __init__(self, predictor, N, d):
        # init particle amount
        self.N = N
        # calculate p(z_t|x_t)
        self.predictor = predictor
        # particle bounds
        self.bounds = (16, 464, 16, 704)
        # dimension of particle matrix
        self.d = d
        # particle matrix
        # column 0, 1 are x, y resp,
        # column 2 is weight
        self.S = initS(N, *self.bounds, d=4)

    def predict(self):
        N = self.S.shape[0]
        self.S[:, 0:2] += normal(0, 5, (N, 2))
        boundS(self.S, *self.bounds)

    def observe(self, frame):
        weights = frame2weights(self.predictor, self.S, frame)
        weights *= self.S[:, 2]
        weights /= np.sum(weights)
        self.S[:, 2] = weights

    def sisr(self, frame):
        self.predict()
        self.observe(frame)
        self.resample()
        return self.S


class SystematicAT(ATBase):
    """
    Systematic Resampling
    """

    def __init__(self, predictor, N=400):
        ATBase.__init__(self, predictor, 20 * N, 3)
        self.N = N

    def resample(self):
        self.S = sys_resample(self.S, self.N)


class ParallelAT(ATBase):
    """
    Multi Particle Filter
    """

    def __init__(self, predictor, N=400, M=40):
        ATBase.__init__(self, predictor, 20 * N, 4)
        self.N = N
        self.S[:, 3] = np.repeat(np.arange(M), 20 * N / M)
        self.M = M

    def resample(self):
        thres = 1e-6
        S = self.S
        Pi = clusterWeight(S[:, 2], S[:, 3], self.M)
        sel = np.nonzero(Pi > thres)[0]
        M = len(sel)
        size = self.N / M
        Sp = np.zeros((size * M, 4))
        for i, m in enumerate(sel):
            S_m = S[S[:,3] == m, :]
        self.S = Sp


class SoftClusteringAT(ATBase):
    """
    Cluster resampling using EM
    """

    def __init__(self, predictor, N=400, M=20):
        ATBase.__init__(self, predictor, 20 * N, 4)
        self.N = N
        # Number of mixtures
        self.M = M
        # clustering
        self.clusterer = KMeans(M, algorithm='full', n_init=5, max_iter=20)
        self.S[:, 3] = np.repeat(np.arange(M), 20 * N / M)

    def cluster(self):
        self.S[:, 3] = self.clusterer.fit_predict(self.S[:, 0:2])

    def resample(self):
        S = self.S
        Sp = np.zeros((self.N, 4))
        Pi = clusterWeight(S[:, 2], S[:, 3], self.M)
        # The number of samples are proportional to cluster weight
        # This eliminates very bad clusters
        mixture_sizes = distribute(Pi, self.N)
        # the base address of the cluster
        base = 0
        for m, size_m in enumerate(mixture_sizes):
            if size_m == 0:
                continue
            # ind = index of particles from mixture m
            ind = (S[:, 3] == m)
            Sm = sys_resample(S[ind, :], size_m)
            Sp[base:base + size_m] = Sm
            base += size_m
        self.S = Sp

    def sisr(self, frame):
        self.predict()
        self.observe(frame)
        self.resample()
        self.cluster()
        return self.S


class HardClusteringAT(ATBase):
    """
    Cluster resampling using lloyd's
    """

    def __init__(self, predictor, N=400, M=20):
        ATBase.__init__(self, predictor, 20 * N, 4)
        self.N = N
        # Number of mixtures
        self.M = M
        self.S[:, 3] = np.repeat(np.arange(M), 20 * N / M)

    def cluster(self):
        score = 720*360
        sampPos = self.S[:, 0:2]
        for i in range(5):
            # Find the best clustering out of 5
            centroids, labels = kmeans2(sampPos, self.M)
            newScore = clusterScore(centroids, labels, sampPos)
            if newScore < score:
                score = newScore
                self.S[:, 3] = labels

    def resample(self):
        S = self.S
        Sp = np.zeros((self.N, 4))
        Pi = clusterWeight(S[:, 2], S[:, 3], self.M)
        # The number of samples are proportional to cluster weight
        # This eliminates very bad clusters
        mixture_sizes = distribute(Pi, self.N)
        # the base address of the cluster
        base = 0
        for m, size_m in enumerate(mixture_sizes):
            if size_m == 0:
                continue
            # ind = index of particles from mixture m
            ind = (S[:, 3] == m)
            Sm = sys_resample(S[ind, :], size_m)
            Sp[base:base + size_m] = Sm
            base += size_m
        self.S = Sp

    def sisr(self, frame):
        self.predict()
        self.observe(frame)
        self.resample()
        self.cluster()
        return self.S
