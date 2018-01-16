import numpy as np
from numpy.random import uniform, normal, randint
from sklearn.cluster import KMeans
from aux_func import normImg
from math import ceil


def initS(M, xmin, xmax, ymin, ymax, d=3):
    S = np.zeros((M, d))
    S[:, 0] = uniform(xmin, xmax, M)
    S[:, 1] = uniform(ymin, ymax, M)
    S[:, 2] = 1. / M
    return S


def boundS(S, xmin, xmax, ymin, ymax):
    """
    Bound S to bounds
    """
    S[S[:, 0] < xmin, 0] = xmin
    S[S[:, 1] < ymin, 1] = ymin
    S[S[:, 0] >= xmax, 0] = xmax - 1e-9
    S[S[:, 1] >= ymax, 1] = ymax - 1e-9


def sys_resample(S, M):
    cdf = create_cdf(S[:, 2])
    D = S.shape[1]
    Sp = np.zeros((M, D))
    i, r = 0, uniform()
    for m in range(M):
        while (m + r) / M > cdf[i]:
            i += 1
        Sp[m, :] = S[i, :]
    Sp[:, 2] = 1. / M
    return Sp


def create_cdf(weights):
    weights = weights / np.sum(weights)
    N = weights.shape[0]
    cdf = np.zeros(N)
    for n in range(N):
        cdf[n] = cdf[n-1] + weights[n]
    return cdf

def clusterWeight(weights, labels, C):
    c_weight = np.zeros(C)
    for c in range(C):
        c_weight[c] = np.sum(weights[labels == c])
    return c_weight

def distribute(weights, N):
    # distribute the samples
    M = weights.shape[0]
    distri = np.zeros(M, dtype=int)
    rollover = 0.
    for c in range(M):
        size = weights[c] * N + rollover
        distri[c] = np.round(size)
        rollover = size - distri[c]
    return distri


OFFSETS = (720 * np.repeat(np.arange(32), 32) +
           np.tile(np.arange(32), 32)).reshape(1, -1)


def frame2weights(predictor, S, frame):
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

    def __init__(self, predictor, N=1000):
        ATBase.__init__(self, predictor, 10 * N, 3)
        self.N = N

    def resample(self):
        self.S = sys_resample(self.S, self.N)


class KMeansAT(ATBase):

    def __init__(self, predictor, N=1000, M=20):
        ATBase.__init__(self, predictor, 10 * N, 4)
        self.N = N
        # Number of mixtures
        self.M = M
        # clustering
        self.clusterer = KMeans(M, algorithm='full',
                                random_state=1, n_init=1, max_iter=10)
        self.S[:, 3] = np.repeat(np.arange(M), 10 * N / M)

    def cluster(self):
        self.S[:, 3] = self.clusterer.fit_predict(self.S[:, 0:2])

    def resample(self):
        S = self.S
        Sp = np.zeros((self.N, 4))
        Pi = clusterWeight(S[:,2], S[:,3], self.M)
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
        Sm[:, 2] = 1. / self.N
        self.S = Sp

    def sisr(self, frame):
        self.predict()
        self.observe(frame)
        self.resample()
        self.cluster()
        return self.S