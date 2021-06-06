#
# Created on Thu May 27 2021 4:24:22 PM
#
# The MIT License (MIT)
# Copyright (c) 2021 Ashwin De Silva
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
# TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# Objective : Graph Learning Methods

# ////// libraries ///// 

# standard 
import numpy as np
from scipy.stats import t

# internal 

# ////// body ///// 

class CorrelationGraphLearn:
    def __init__(self, X, alpha):
        """
        Learning graph topolgy using correlation coefficients estimated 
        from nodal samples.

        Math
        ----------
        a_ij = |r_ij|,   when rho_ij is significantly non-zero
        a_ij = 0,        when rho_ij is significantly zero

        Parameters
        ----------
        X : numpy array
            2 x 2 matrix with each row representing a channel and each
            column representing a time point/ sample.

        alpha : float scalar
            significant level used when testing the staistical significance
            of the computed sample correlation coefficients
        """
        super(CorrelationGraphLearn, self).__init__()
        self.X = X
        self.alpha = alpha
        self.L = X.shape[0] # num channels
        self.N = X.shape[1] # num samples / time points

    def isRhoSignificant(self, r):
        """
        staitsical test for testing the significance of the correlation
        coefficient.

        Parameters
        ----------
        r : float 
            The computed sample correlation coefficient

        Returns
        ----------
        Whether r is significant or not : bool
        """
        n = self.N
        df = n - 2
        t_test = r*np.sqrt(n-2)/np.sqrt(1-r**2)
        t_alpha = t.ppf(self.alpha/2, df)
        if abs(t_test) > abs(t_alpha):
            return True
        else:
            return False

    def findGraph(self):
        """
        Learns and returns the graph topology in the form of an
        adjacency matrix

        Returns
        ----------
        W : numpy array of shape (nChannels, nChannels)
        """
        Y = self.X
        L = self.L
        N = self.N
        W = np.zeros((L, L))
        Ybar = np.mean(Y, axis=1, keepdims=True)*np.ones((L, N))
        for i in range(L):
            for j in range(L):
                r_ij = np.inner(Y[i] - Ybar[i], Y[j] - Ybar[j])
                r_ij /= np.linalg.norm(Y[i] - Ybar[i], ord=2) * np.linalg.norm(Y[j] - Ybar[j], ord=2)
                if self.isRhoSignificant(r_ij): # statistical significance test
                    W[i, j] = abs(r_ij)
                else:
                    W[i, j] = 0
        return W
          
class PartialCorrelationGraphLearn:
    def __init__(self, X, alpha):
        """
        Learning graph topology using partial correlation coefficients
        estimated using nodal samples.
        """
        super(PartialCorrelationGraphLearn, self).__init__()
        self.X = X
        self.alpha = alpha
        self.L = X.shape[0] # num channels
        self.N = X.shape[1] # num samples / time points

    def isRhoBarSignificant(self, rbar):
        """
        staitsical test for testing the significance of the partial correlation
        coefficient.

        Parameters
        ----------
        rbar : float 
            The computed sample partial correlation coefficient

        Returns
        ----------
        Whether rbar is significant or not : bool
        """
        n = self.N
        k = self.L - 2
        df = n - 2 - k
        t_test = rbar*np.sqrt(df/(1-rbar**2))
        t_alpha = t.ppf(self.alpha/2, df)
        if abs(t_test) > abs(t_alpha):
            return True
        else:
            return False

    def findGraph(self):
        """
        Learns and returns the graph topology in the form of an
        adjacency matrix

        Returns
        ----------
        W : numpy array of shape (nChannels, nChannels)
        """
        Y = self.X
        L = self.L
        Theta_inv = np.cov(Y)
        Theta = np.linalg.inv(Theta_inv)
        W = np.zeros((L, L))
        for i in range(L):
            for j in range(L):
                rbar_ij = - Theta[i, j]/ np.sqrt(Theta[i, i] * Theta[j, j])
                if self.isRhoBarSignificant:    # statistical significance test
                    W[i, j] = abs(rbar_ij)
                else:
                    W[i, j] = 0
        return W

class SmoothSignalGraphLearn:
    """
    Primal dual algorithm for learning graphs on smooth signals
    """
    def __init__(self, X, alpha, beta, gamma, epsilon=1e-4, imax=2000, verbosity=True):
        super(SmoothSignalGraphLearn, self).__init__()
        self.X = X
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        self.imax = imax
        self.verbosity = verbosity

        self.m = X.shape[0]
        self.n = int(self.m*(self.m-1)/2)

    def getPairwiseDistanceMatrix(self):
        """
        compute the pairwise distance matrix $Z \in \mathbb{R}^{m \times m}$ 
        of the observations matrix $X \in \mathbb{R}^{m \times n}$ as defined below.
        $$ Z_{i, j} = || x_i - x_j ||^2 $$
        """
        X = self.X
        m = self.m
        Z = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                Z[i, j] = np.linalg.norm(X[i] - X[j], ord=2)**2 # using l-2 norm
        return Z

    def vectorize(self, Y):
        """
        Form an array $y \in \mathbb{m(m-1)/2}$ that only contains the upper 
        traingle indices of the input matrix $Y \in \mathbb{R}^{m \times m}$
        """
        m = self.m
        idx = np.triu_indices(m, 1)
        return Y[idx]

    def unvectorize(self, y):
        """
        Form a symmetrical matrix $Y \in \mathbb{R}^{m \times m}$ where
        diag(Y) = 0 given an array $y \in \mathbb{m(m-1)/2}$
        """
        m = self.m
        Y = np.zeros((m, m))
        i, j = np.triu_indices(m, 1)
        Y[i, j] = y
        Y[j, i] = y
        return Y

    def getLinearOperator(self):
        m = self.m
        n = int(m*(m-1)/2)
        S = np.zeros((m, n))
        start = 0
        for i in range(m-1):
            end = start + (m-i-1)
            S[i, start:end] = 1
            S[i+1:, start:end] = np.identity(m-i-1)
            start = end
        return S

    def findGraph(self):
        """
        Run the primal dual algorithm for the following model : 
        $$ \min_{W \in \mathcal{W}_m} || W \circ X ||_{1, 1} - \alpha \bm{1}^\top \log (W \bm{1}) + \beta ||W||_F^2 $$
        """
        # vectorized pairwise distance matrix
        Z = self.getPairwiseDistanceMatrix()
        z = self.vectorize(Z)
        z = np.expand_dims(z, -1)

        # Linear Operator
        S = self.getLinearOperator()

        # hyperparameters
        alpha = self.alpha
        beta = self.beta
        gamma = self.gamma
        epsilon = self.epsilon
        imax = self.imax
    
        # initializations
        wi = np.zeros((self.n, 1))          # zero initialization
        di = np.zeros((self.m, 1))          # zero initialization

        # primal dual algorithm
        for i in range(imax):
            yi = wi - gamma*(2*beta*wi + np.matmul(S.T,di))
            yi_bar = di + gamma*np.matmul(S, wi)
            pi = np.maximum(0, yi - 2*gamma*z)
            pi_bar = (yi_bar - np.sqrt(yi_bar**2 + 4*alpha*gamma))/2
            qi = pi - gamma*(2*beta*pi + np.matmul(S.T, pi_bar)) 
            qi_bar = pi_bar + gamma*np.matmul(S, pi) 
            wii = wi - yi + qi
            dii = di - yi_bar + qi_bar
            w_ratio = np.linalg.norm(wii - wi, ord=2)/ np.linalg.norm(wi, ord=2)
            d_ratio = np.linalg.norm(dii - di, ord=2)/ np.linalg.norm(di, ord=2)
            wi = wii
            di = dii
            if self.verbosity:
                print("iter : {:n}, w_ratio : {:.3f}, d_ratio : {:.3f}".format(i, w_ratio, d_ratio))
            if w_ratio < epsilon and d_ratio < epsilon:
                break

        # Adjacency matrix
        W = self.unvectorize(wii.squeeze())
        return W

class AutoregressGraphLearn:
    """
    Sparse Autoregressive Modelling based Graph Learning (need modification f(x) + g(Lx))
    """
    def __init__(self, X, beta, gamma, eta, epsilon=1e-4, imax=2000, verbosity=True):
        super(AutoregressGraphLearn, self).__init__()
        self.X = X                      # multi-channel siganl
        self.beta = beta                # regularization parameter (promotes sparsity)
        self.gamma = gamma              # step size
        self.eta = eta                  # value in [0, 1]
        self.epsilon = epsilon          # tolerance
        self.imax = imax                # maximum iteration
        self.verbosity = verbosity      

        self.L = X.shape[0]                     # number of channels / number of nodes of the learned graph
        self.N = X.shape[1]                   # number of multi-channel samples
        self.K = int(self.L*(self.L-1)/2)       # size of the Vec(W) | size of the upper triangle w.o. the main diagonal

    def vectorize(self, Y):
        """
        Form an array $y \in \mathbb{L(L-1)/2}$ that only contains the upper 
        traingle indices of the input matrix $Y \in \mathbb{R}^{L \times L}$
        """
        L = self.L
        idx = np.triu_indices(L, 1)
        return Y[idx]

    def unvectorize(self, y):
        """
        Form a symmetrical matrix $Y \in \mathbb{R}^{m \times m}$ where
        diag(Y) = 0 given an array $y \in \mathbb{m(m-1)/2}$
        """
        L = self.L
        Y = np.zeros((L, L))
        i, j = np.triu_indices(L, 1)
        Y[i, j] = y
        Y[j, i] = y
        return Y

    def getLinearOperator(self):
        """
        Get the linear operator S s.t. W 1 = S w
        Here, w \in R^{L*(L-1)/2} 
        """
        L = self.L
        K = self.K
        S = np.zeros((L, K))
        start = 0
        for i in range(L-1):
            end = start + (L-i-1)
            S[i, start:end] = 1
            S[i+1:, start:end] = np.identity(L-i-1)
            start = end
        return S

    def ConvertVec2Mat(self, x, S):
        """
        Arrange in the provided vector within the linear operator S
        """
        L = self.L
        K = self.K
        for i in range(L):
            z = np.delete(x, i)
            for j in range(K):
                if S[i, j] == 1:
                    S[i, j] = z[0]
                    z = np.delete(z, 0)
        return S

    def softThresholding(self, x, psi):
        """
        Compute the output of the Soft Thresholding function defined below :
        SFT(x_i) = x_i - psi if x_i >= psi
        STF(x_i) = 0         if |x_i| <= psi
        STF(x_i) = x_i + psi if x_i <= -psi
        This is an element-wise operation :
        STF(x) = sign(x) maximum(|x| - psi, 0)
        """
        return np.sign(x)*np.maximum(np.abs(x)-psi, 0)

    def getProxOp(self, x, S):
        """
        returns A^{-1}(gamma) b(u, gamma)
        """
        Y = self.X.T
        L = self.L
        N = self.N
        K = self.K
        gamma = self.gamma

        # compute A and b
        A = np.zeros((K, K))
        b = x
        for n in range(1, N):
            Sn_1 = self.ConvertVec2Mat(Y[n-1], S)
            A += gamma * np.matmul(Sn_1.T, Sn_1) + 1/(N-1) * np.identity(K)
            b += gamma * np.matmul(Sn_1.T, Y[n].reshape(L, 1))
        return np.maximum(0, np.matmul(np.linalg.inv(A), b))

    def findGraph(self):
        """
        Run the optimization algorithm for the following model : 
        $$ min 1/2 sum_{n=1}^{N-1} || y[n] - S[n-1]w||^2 + beta * ||w||_1 $$
        """
        # Data / Obseravation Matrix
        X = self.X
        L = self.L
        N = self.N
        K = self.K

        # Linear Operator
        S = self.getLinearOperator()

        # hyperparameters
        beta = self.beta
        gamma = self.gamma
        eta = self.eta
        epsilon = self.epsilon
        imax = self.imax
    
        # initializations
        wi = np.zeros((self.K, 1))          # zero initialization

        # optimization algorithm
        for i in range(imax):
            yi = self.softThresholding(wi, beta*gamma)
            # mui = np.random.rand(1)*(2 - 2*eta) + eta
            mui = (1 + eta)/2
            wii = wi + mui*(self.getProxOp(2*yi-wi, S) - yi)
            w_ratio = np.linalg.norm(wii - wi, ord=2)/ np.linalg.norm(wi, ord=2)
            wi = wii
            if self.verbosity:
                print("iter : {:n}, w_ratio : {:.3f}".format(i, w_ratio))
            if w_ratio < epsilon:
                break

        # Adjacency matrix
        W = self.unvectorize(wii.squeeze())
        return W

class SmoothAutoregressGraphLearn:
    """
    Sparse Autoregressive Modelling based Graph Learning on Smooth Signals
    """
    def __init__(self, X, beta, gamma, epsilon=1e-4, imax=2000, verbosity=True):
        super(SmoothAutoregressGraphLearn, self).__init__()
        self.X = X                      # multi-channel siganl
        self.beta = beta                # regularization parameter (promotes sparsity)
        self.gamma = gamma              # step size
        self.epsilon = epsilon          # tolerance
        self.imax = imax                # maximum iteration
        self.verbosity = verbosity      

        self.L = X.shape[0]                     # number of channels / number of nodes of the learned graph
        self.N = X.shape[1]                     # number of multi-channel samples
        self.K = int(self.L*(self.L-1)/2)       # size of the Vec(W) | size of the upper triangle w.o. the main diagonal

    def vectorize(self, Y):
        """
        Form an array $y \in \mathbb{L(L-1)/2}$ that only contains the upper 
        traingle indices of the input matrix $Y \in \mathbb{R}^{L \times L}$
        """
        L = self.L
        idx = np.triu_indices(L, 1)
        return Y[idx]

    def unvectorize(self, y):
        """
        Form a symmetrical matrix $Y \in \mathbb{R}^{m \times m}$ where
        diag(Y) = 0 given an array $y \in \mathbb{m(m-1)/2}$
        """
        L = self.L
        Y = np.zeros((L, L))
        i, j = np.triu_indices(L, 1)
        Y[i, j] = y
        Y[j, i] = y
        return Y

    def getPairwiseDistanceMatrix(self):
        """
        compute the pairwise distance matrix $Z \in \mathbb{R}^{m \times m}$ 
        of the observations matrix $X \in \mathbb{R}^{m \times n}$ as defined below.
        $$ Z_{i, j} = || x_i - x_j ||^2 $$
        """
        X = self.X
        L = self.L
        Z = np.zeros((L, L))
        for i in range(L):
            for j in range(L):
                Z[i, j] = np.linalg.norm(X[i] - X[j], ord=2)**2 # using l-2 norm
        return Z

    def getArrangedTimeSeriesData(self):
        """
        Obtain the vector y and matrix S from X
        y = [y[1]' y[2]' ... y[N-1]']'
        S = [S[0]' S[1]' ... S[N-2]']'
        """
        X = self.X
        K = self.K
        N = self.N

        # obtain y
        y = X[:, 1:].reshape(-1, order='F')

        # obtain S
        So = self.getLinearOperator()
        S = np.empty((0, K))
        for n in range(N-1):
            yn = X[:, n]
            Sn = self.ConvertVec2Mat(yn, So)
            S = np.vstack((S, Sn))
        
        return y, S

    def getLinearOperator(self):
        """
        Get the linear operator S s.t. W 1 = S w
        Here, w \in R^{L*(L-1)/2} 
        """
        L = self.L
        K = self.K
        S = np.zeros((L, K))
        start = 0
        for i in range(L-1):
            end = start + (L-i-1)
            S[i, start:end] = 1
            S[i+1:, start:end] = np.identity(L-i-1)
            start = end
        return S

    def ConvertVec2Mat(self, x, S):
        """
        Arrange in the provided vector within the linear operator S
        """
        L = self.L
        K = self.K
        for i in range(L):
            z = np.delete(x, i)
            for j in range(K):
                if S[i, j] == 1:
                    S[i, j] = z[0]
                    z = np.delete(z, 0)
        return S

    def findGraph(self):
        """
        Run the optimization algorithm for the following model : 
        $$ min 1/2 ||W o Z||_1 + 1/2 sum_{n=1}^{N-1} || y[n] - Wy[n-1]||^2 + beta * ||W||_2^2 $$
        """
        # get y and S (from Time Series X)
        y, S = self.getArrangedTimeSeriesData()
        y = y.reshape(y.shape[0], 1)

        # get vectorized pairwise distance matrix
        Z = self.getPairwiseDistanceMatrix()
        z = self.vectorize(Z)
        z = z.reshape(z.shape[0], 1)

        # dimensions
        L = self.L
        N = self.N
        K = self.K

        # hyperparameters
        beta = self.beta
        gamma = self.gamma
        epsilon = self.epsilon
        imax = self.imax
    
        # initializations
        wi = np.zeros((K, 1))          # zero initialization
        ui = np.zeros((L*(N-1), 1))

        # optimization algorithm
        for i in range(imax):
            v1i = wi - gamma*(2*beta*wi + np.matmul(S.T, ui))
            v2i = ui + gamma*np.matmul(S, wi)
            p1i = np.maximum(0, v1i - 2*gamma*z)
            p2i = (v2i - gamma*y)/(gamma + 1)
            q1i = p1i - gamma*(2*beta*p1i + np.matmul(S.T, p2i))
            q2i = p2i + gamma*np.matmul(S, p1i)
            wii = wi - v1i + q1i
            uii = ui - v2i + q2i 
            w_ratio = np.linalg.norm(wii - wi, ord=2)/ np.linalg.norm(wi, ord=2)
            u_ratio = np.linalg.norm(uii - ui, ord=2)/ np.linalg.norm(ui, ord=2)
            wi = wii
            ui = uii
            if self.verbosity:
                print("iter : {:n}, w_ratio : {:.4f}, u_ratio : {:4f}".format(i, w_ratio, u_ratio))
            if w_ratio < epsilon and u_ratio < epsilon:
                break

        # Adjacency matrix
        W = self.unvectorize(wi.squeeze())
        return W

