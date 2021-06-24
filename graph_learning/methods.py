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
from numpy import linalg
from numpy.lib.index_tricks import fill_diagonal
from scipy.stats import t
from scipy.linalg import khatri_rao, eig
from cvxopt import matrix, solvers
from optimizers.l1regls import l1regls
from optimizers.l1 import l1

# internal 

# ////// body ///// 

class CorrelationGraphLearn:
    """ Learns graph topolgy using correlation coefficients estimated 
        from nodal samples.

        Parameters
        ----------
        X : numpy array
            array of shape (L, N) where each row representing a channel and each
            column representing a time point/sample.
        alpha : float
            significant level used when testing the staistical significance
            of the population correlation coefficients
        """
    def __init__(self, X, alpha):
        super(CorrelationGraphLearn, self).__init__()
        self.X = X
        self.alpha = alpha
        self.L = X.shape[0] # num channels
        self.N = X.shape[1] # num samples / time points

    def isRhoSignificant(self, r):
        """
        Carries out the staitsical test to test the significance of the population
        correlation coefficient.

        Parameters
        ----------
        r : float 
            The sample correlation coefficient

        Returns
        ----------
        bool
            If significant, returns ``True``. If not significant, returns ``False``
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
        Computes the adjacency matrix.

        Returns
        ----------
        numpy array
            An adjacency matrix of shape (L, L)
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
    def __init__(self, X, alpha=0.5):
        """
        Learns graph topolgy using partial correlation coefficients estimated 
        from nodal samples.

        Parameters
        ----------
        X : numpy array
            array of shape (L, N) where each row representing a channel and each
            column representing a time point/sample.
        alpha : float
            significant level used when testing the staistical significance
            of the population partial correlation coefficients, by default 0.5
        """
        super(PartialCorrelationGraphLearn, self).__init__()
        self.X = X
        self.alpha = alpha
        self.L = X.shape[0] # num channels
        self.N = X.shape[1] # num samples / time points

    def isRhoBarSignificant(self, rbar):
        """
        Carries out the staitsical test for testing the significance of the population 
        partial correlation coefficient.

        Parameters
        ----------
        rbar : float 
            The sample partial correlation coefficient

        Returns
        ----------
        bool
            If significant, returns ``True``. If not significant, returns ``False``
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
        Computes the adjacency matrix.

        Returns
        ----------
        numpy array
            An adjacency matrix of shape (L, L)
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

class GraphicalLassoGraphLearn:
    """Learns graph topolgy using sparse inverse covariance estimation using 
    the graphical Lasso algorithm implemented by ADMM.

    Parameters
    ----------
    X : numpy array
        array of shape (L, N) where each row representing a channel and each
        column representing a time point/sample.
    beta : float
        Regularization parameter (sparsity tuner)
    gamma : float
        step size
    imax : int, optional
        maximum number of iterations, by default 2000
    epsilon : float, optional
        tolarence, by default 1e-4
    alpha : float, optional
        significant level used when testing the staistical significance
        of the population partial correlation coefficients, by default 0.05
    verbosity : bool, optional
        print the convergence indicator at current iteration, by default False
    """
    def __init__(self, X, beta, gamma, imax=2000, epsilon=1e-4, alpha=0.05, verbosity=False):
        super(GraphicalLassoGraphLearn, self).__init__()
        self.X = X                      # multi-channel window
        self.beta = beta                # regularization parameters (sparsity tuner)
        self.gamma = gamma              # step size
        self.imax = imax                # maximum number of iterations
        self.epsilon = epsilon          # tolerance
        self.alpha = alpha              # statistical significance level
        self.verbosity = verbosity

        self.L = X.shape[0]             # number of channels/nodes
        self.N = X.shape[1]             # number of time points/samples

    def getSampleCovMatrix(self):
        """Obtains the sample covariance matrix from the nodal samples

        Returns
        -------
        array
            Sample covariance matrix of size (L, L)
        """
        return np.matmul(self.X - np.mean(self.X, axis=-1, keepdims=True), (self.X - np.mean(self.X, axis=-1, keepdims=True)).T)/ self.N

    def getFgamma(self, Z):
        """Computes the function in ADMM that includes the eigendecomposition

        Parameters
        ----------
        Z : array
            input matrix to the function of size (L, L)

        Returns
        -------
        array
            output of the function of size (L, L)
        """
        lambda_i, U = np.linalg.eig(Z)
        Lamda_prime = np.diag(lambda_i + np.sqrt(lambda_i**2 + 4/self.gamma))
        return 0.5 * np.matmul(np.matmul(U, Lamda_prime), U.T) 

    def softThresholding(self, x, zeta):
        """Computes the elementwise soft-thresholding output

        Parameters
        ----------
        x : array
            input array
        zeta : float
            parameter of the soft-thresholding function

        Returns
        -------
        array
            soft-thresholded vector/matrix
        """
        return np.sign(x)*np.maximum(np.abs(x)-zeta, 0)

    def isRhoBarSignificant(self, rbar):
        """
        Carries out the staitsical test for testing the significance of the population 
        partial correlation coefficient.

        Parameters
        ----------
        rbar : float 
            The sample partial correlation coefficient

        Returns
        ----------
        bool
            If significant, returns ``True``. If not significant, returns ``False``
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
        Computes the adjacency matrix.

        Returns
        ----------
        numpy array
            An adjacency matrix of shape (L, L)
        """
        # get sample covariance matrix
        Sigma_hat = self.getSampleCovMatrix()

        # dimensions
        L = self.L

        # hyperparameters
        beta = self.beta
        gamma = self.gamma
        epsilon = self.epsilon
        imax = self.imax

        # initializations
        Lambda_i = np.zeros((L, L))
        Psi_i = np.zeros((L, L))
        Theta_i = np.linalg.inv(Sigma_hat + beta*np.identity(L))

        # optimization
        for i in range(imax):
            Theta_ii = self.getFgamma(Psi_i - Lambda_i/gamma - Sigma_hat/gamma)
            Psi_ii = self.softThresholding(Theta_ii - Lambda_i/gamma, beta/gamma)
            Lambda_ii = Lambda_i + gamma * (Theta_ii - Psi_ii)
            Theta_diff = np.linalg.norm(Theta_ii - Theta_i, ord='fro')/(np.linalg.norm(Theta_i, ord='fro') + 1e-8)
            Theta_i = Theta_ii
            Psi_i = Psi_ii
            Lambda_i = Lambda_ii
            if self.verbosity:
                print("iter : {:n}, Theta_ratio : {:.3f}".format(i, Theta_diff))
            if Theta_diff < epsilon:
                break
        Theta = Theta_i

        # obtain graph topology
        W = np.zeros((L, L))
        for i in range(L):
            for j in range(L):
                rbar_ij = - Theta[i, j]/ np.sqrt(Theta[i, i] * Theta[j, j])
                if self.isRhoBarSignificant:    # statistical significance test
                    W[i, j] = abs(rbar_ij)
                else:
                    W[i, j] = 0
        return W

class StructualEqnModelGraphLearn:
    """Learns graph topolgy using structual equation modelling. (Fix the error in ADMM)

    Parameters
    ----------
    X : numpy array
        array of shape (L, N) where each row representing a channel and each
        column representing a time point/sample.
    beta : float
        Regularization parameter (sparsity tuner)
    gamma : float
        step size
    imax : int, optional
        maximum number of iterations, by default 2000
    epsilon : float, optional
        tolarence, by default 1e-4
    verbosity : bool, optional
        print the convergence indicator at current iteration, by default False
    """
    def __init__(self, X, beta, gamma, imax=2000, epsilon=1e-4, verbosity=False):
        super(StructualEqnModelGraphLearn, self).__init__()
        self.X = X                      # multi-channel window
        self.beta = beta                # regularization parameters (sparsity tuner)
        self.gamma = gamma              # step size
        self.imax = imax                # maximum number of iterations
        self.epsilon = epsilon          # tolerance
        self.verbosity = verbosity

        self.L = X.shape[0]             # number of channels/nodes
        self.N = X.shape[1]             # number of time points/samples

    def softThresholding(self, x, zeta):
        """Computes the elementwise soft-thresholding output

        Parameters
        ----------
        x : array
            input array
        zeta : float
            parameter of the soft-thresholding function

        Returns
        -------
        array
            soft-thresholded vector/matrix
        """
        return np.sign(x)*np.maximum(np.abs(x)-zeta, 0)

    def findGraph(self):
        """
        Computes the adjacency matrix.

        Returns
        ----------
        numpy array
            An adjacency matrix of shape (L, L)
        """
        # get X
        X = self.X

        # dimensions
        L = self.L

        # hyperparameters
        beta = self.beta
        gamma = self.gamma
        epsilon = self.epsilon
        imax = self.imax

        # initializations
        Wi = np.random.rand(L, L)
        Zi = np.random.rand(L, L)

        # optimization
        for i in range(imax):
            Ui = np.matmul(gamma * np.matmul(X, X.T) + Wi - Zi, np.linalg.inv(gamma * np.matmul(X, X.T) + np.identity(L)))
            Wii = self.softThresholding(Ui + Zi, beta/gamma)
            Zii = Zi + Ui - Wii
            W_diff = np.linalg.norm(Wii - Wi, 'fro')/(np.linalg.norm(Wi, 'fro') + 1e-8)
            Z_diff = np.linalg.norm(Zii - Zi, 'fro')/(np.linalg.norm(Zi, 'fro') + 1e-8)
            Wi = Wii
            Zi = Zii
            if self.verbosity:
                print("iter : {:n}, W_ratio : {:.3f}, Z_ratio : {:.3f}".format(i, W_diff, Z_diff))
            if W_diff < epsilon and Z_diff < epsilon:
                break
        W = Wi
        W[W < 1e-5] = 0
        return W

class SVARMGraphLearn:
    def __init__(self, X, p, beta, gamma, imax=2000, epsilon=1e-4, verbosity=False):
        super(SVARMGraphLearn, self).__init__()
        self.X = X                      # multi-channel window
        self.p = p                      # autogressive model order
        self.beta = beta                # regularization parameters (sparsity tuner)
        self.gamma = gamma              # step size
        self.imax = imax                # maximum number of iterations
        self.epsilon = epsilon          # tolerance
        self.verbosity = verbosity

        self.L = X.shape[0]             # number of channels/nodes
        self.N = X.shape[1]             # number of time points/samples

    def getArrangedTimeSeries(self):
        N = self.N
        L = self.L
        p = self.p
        X = np.empty((N-p, 0))
        for i in range(L):
            yi = self.X[i]
            Xi = np.zeros((N-p, p))
            for k in range(Xi.shape[0]):
                Xi[k] = np.flip(yi[k:k+p]).reshape(1, p)
            X = np.hstack((X, Xi))
        return X

    def groupSoftThresholding(self, x, zeta):
        return np.maximum(0, 1 - zeta/np.linalg.norm(x, ord=2)) * x

    def SmartGlassoEstimator(self, X, yi):
        # dimensions
        L = self.L
        p = self.p

        # hyperparameters
        beta = self.beta
        gamma = self.gamma
        epsilon = self.epsilon
        imax = self.imax

        # initializations
        wk = 0.01*np.ones((L*p, 1))
        zk = 0.01*np.ones((L*p, 1))
        uk = 0.01*np.ones((L*p, 1))

        # optimization
        for k in range(imax):
            wkk = np.matmul(np.linalg.inv(np.matmul(X.T, X) + gamma*np.identity(L*p)), np.matmul(X.T, yi) + gamma*(zk - uk)) 
            zkk = np.zeros((L*p, 1))
            for j in range(L):
                zkk[j*p:j*p+p] = self.groupSoftThresholding(wkk[j*p:j*p+p] + uk[j*p:j*p+p], beta/gamma)
            ukk = uk + wkk - zkk
            w_ratio = np.linalg.norm(wkk - wk, 2)/(np.linalg.norm(wk, 2) + 1e-7)
            z_ratio = np.linalg.norm(zkk - zk, 2)/(np.linalg.norm(zk, 2) + 1e-7)
            u_ratio = np.linalg.norm(ukk - uk, 2)/(np.linalg.norm(uk, 2) + 1e-7)
            wk = wkk
            zk = zkk
            uk = ukk
            if self.verbosity:
                print("iter : {:n}, w_ratio : {:.3f}, z_ratio : {:.3f}, u_ratio : {:.3f}".format(k, w_ratio, z_ratio, u_ratio))
            if w_ratio < epsilon and z_ratio < epsilon and u_ratio < epsilon:
                break
        return wk

    def findGraph(self):
        # get y and S (from Time Series X)
        X = self.getArrangedTimeSeries()

        # dimesions
        N = self.N
        L = self.L
        p = self.p

        # optimize
        Wp = np.zeros((L, L, p))
        for i in range(L):
            yi = self.X[i][p:].reshape(N-p, 1)
            # print("Estimatin Coefficients of Node : {}".format(i))
            wi = self.SmartGlassoEstimator(X, yi)
            Wp[i, :, :] = wi.reshape(L, p)
        Wp[abs(Wp) < 1e-3] = 0    

        # get graph topology
        W = np.zeros((L, L))
        for i in range(L):
            for j in range(L):
                k = Wp[i, j, :]
                if Wp[i, j, :].all():
                    W[i, j] = 1
                else:
                    continue
        return Wp

class SmoothSignalGraphLearn:
    """Executes the smooothness based graph learning algorithm proposed by 
        Kalofolias et al (2016).

        Parameters
        ----------
        X : numpy array
            Multi-channel signal window of shape (num_channels, num_samples)
        alpha : float
            The parameter that regulates the overall connectivity of the graph. Bigger the
            alpha, better the connectivity.
        beta : float
            The parameter that regulates the density of the graph. Bigger the beta, the denser
            the graph
        gamma : float
            Step size of the iterative algorithm
        epsilon : tolerance, optional
            Tolerance that determines the termination of the iterative algorithm, by default 1e-4
        imax : int, optional
            maximum number of iterations, by default 2000
        verbosity : bool, optional
            If True, the conergence parameters would be printed, by default True
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
        self.N = X.shape[1]

    def getPairwiseDistanceMatrix(self):
        """Computes the pairwise distances matrix Z from X.

        Returns
        -------
        Z : numpy array
            The pairwise distances matrix of the multi-channel signal window.
        """
        X = self.X
        m = self.m
        Z = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                Z[i, j] = np.linalg.norm(X[i] - X[j], ord=2)**2 # using l-2 norm
        return Z

    def getPairwiseSecantDistanceMatrix(self):
        """Computes the pairwise secant distances matrix Z from X.

        Returns
        -------
         Z : numpy array
            The pairwise secant distances matrix of the multi-channel signal window.
        """
        X = self.X
        m = self.m
        Z = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                Z[i, j] = np.linalg.norm(X[i])*np.linalg.norm(X[j]) / np.dot(X[i], X[j])
        return Z

    def vectorize(self, Y):
        """Form a vector that contains only the elements of the upper triangle (exclduing
        the main diagonal) of a matrix

        Parameters
        ----------
        Y : numpy array
            Input square matrix

        Returns
        -------
        numpy array
            Vector that contains only the elements of the upper triangle (exclduing
        the main diagonal) of Y
        """
        m = self.m
        idx = np.triu_indices(m, 1)
        return Y[idx]

    def unvectorize(self, y):
        """Given a vector that contains only the elements of the upper triangle (excluding 
        the main diagonal), this function obtains the corresponding matrix

        Parameters
        ----------
        y : numpy array
            Vector that contains only the elements of the upper triangle (exclduing
        the main diagonal) of Y

        Returns
        -------
        Y : numpy array
            Corresponding matrix of y
        """
        m = self.m
        Y = np.zeros((m, m))
        i, j = np.triu_indices(m, 1)
        Y[i, j] = y
        Y[j, i] = y
        return Y

    def getLinearOperator(self):
        """Obtains the linear operator S such that W 1 = S w.

        Returns
        -------
        S : numpy array
            The linear operator which satistifies W 1 = S w
        """
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
        """Runs the primal-dual algorithm to solve the optimization problem stated in
        Kalofolias et al (2016)

        Returns
        -------
        W : numpy array
            Learnt adjacency matrix of the multi-channel signal window
        """
        # vectorized pairwise distance matrix
        Z = self.getPairwiseDistanceMatrix()
        # Z = self.getPairwiseSecantDistanceMatrix()
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
            w_ratio = np.linalg.norm(wii - wi)/ np.linalg.norm(wi)
            d_ratio = np.linalg.norm(dii - di)/ np.linalg.norm(di)
            wi = wii
            di = dii
            if self.verbosity:
                print("iter : {:n}, w_ratio : {:.3f}, d_ratio : {:.3f}".format(i, w_ratio, d_ratio))
            if w_ratio < epsilon and d_ratio < epsilon:
                break

        # Adjacency matrix
        W = self.unvectorize(wii.squeeze())
        W[W < 1e-5] = 0
        return W

class AutoregressGraphLearn:
    """
    Vector Autoregressive Modelling based Graph Learning (solved using Forward
    - Backward Splitting)
    """
    def __init__(self, X, beta, gamma, delta, eta=0.01, epsilon=1e-4, imax=2000, verbosity=True):
        super(AutoregressGraphLearn, self).__init__()
        self.X = X                      # multi-channel siganl
        self.beta = beta                # regularization parameter (promotes sparsity)
        self.eta = eta                  # value in [0, 1] (check the vector norm of S before selecting)
        self.gamma = gamma              # step size
        self.delta = delta              # another step
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

    def getArrangedTimeSeriesData(self):
        """
        Obtain the vector y and matrix S from X
        y = [y[1]' y[2]' ... y[N-1]']'
        S = [S[0]' S[1]' ... S[N-2]']'
        Check this function!!!
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

    def findGraph(self):
        """
        Run the optimization algorithm for the following model : 
        $$ min 1/2 sum_{n=1}^{N-1} || y[n] - S[n-1]w||^2 + beta * ||w||_1 $$
        """
        # get y and S (from Time Series X)
        y, S = self.getArrangedTimeSeriesData()
        y = y.reshape(y.shape[0], 1)

        # hyperparameters
        beta = self.beta
        gamma = self.gamma
        delta = self.delta
        eta = self.eta
        epsilon = self.epsilon
        imax = self.imax
    
        # initializations
        wi = np.zeros((self.K, 1))          # zero initialization
        # gamma = 1 / np.linalg.norm(S, ord=2)
        # delta = (1 + eta)/2

        # optimization algorithm
        for i in range(imax):
            yi = wi - gamma * np.matmul(S.T, (y - np.matmul(S, wi)))
            wii = wi + delta * ( self.softThresholding(yi, gamma*beta) - wi)
            w_ratio = np.linalg.norm(wii - wi, ord='fro')/ np.linalg.norm(wi, ord='fro')
            wi = wii
            if self.verbosity:
                print("iter : {:n}, w_ratio : {:.3f}".format(i, w_ratio))
            if w_ratio < epsilon:
                break

        # Adjacency matrix
        W = self.unvectorize(wii.squeeze())
        W = abs(W)
        W[W < 1e-5] = 0
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

class DiffusionGraphLearn:
    def __init__(self, X, beta_1, p, beta_2, verbosity=True):
        super(DiffusionGraphLearn, self).__init__()
        self.X = X
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.p = p
        self.verbosity = verbosity

        self.L = X.shape[0]
        self.T = X.shape[1]
        self.K = int(self.L*(self.L-1)/2)

    def unvectorize(self, y):
        """
        Form a symmetrical matrix $Y \in \mathbb{R}^{m \times m}$ where
        diag(Y) = 0 given an array $y \in \mathbb{m(m-1)/2}$
        """
        L = self.L
        Y = np.zeros((L, L))
        i, j = np.triu_indices(L, 1)
        Y[i, j] = y.squeeze()
        Y[j, i] = y.squeeze()
        return Y

    def getArrangedTimeSeriesData(self):
        """
        Obtain the vector y and matrix S from X
        X_t = [x_p' x_p+1' ... x_T-1']'
        S_t-r = [S_p-r' S_p-r+1' ... S_T-1-r']'
        S_1p = [S_t-1 S_t-2 ... S_t-p]
        Y_t-r = [ diag(x_p-r) diag(x_p-r+1) ... diag(x_T-r+1)]'
        Y_1p = [ Y_t-1 Y_t-2 ... Y_t-p]
        """
        X = self.X
        K = self.K
        T = self.T
        L = self.L
        p = self.p

        # obtain X_t
        X_t = np.expand_dims(X[:, p:].reshape(-1, order='F'), -1)

        # obtain Y_1p
        Y_1p = np.empty((L*(T-p), 0))
        for r in range(1, p+1):
            Yr = np.empty((0, L))
            for t in range(p-r, T-r):
                xt = X[:, t]
                Yr = np.vstack((Yr, np.diag(xt)))
            Y_1p = np.hstack((Y_1p, Yr))

        # obtain S_1p
        S_1p = np.empty((L*(T-p), 0))
        for r in range(1, p+1):
            Sr = np.empty((0, K))
            for t in range(p-r, T-r):
                xt = X[:, t]
                St = self.ConvertVec2Mat(xt, self.getLinearOperator())
                Sr = np.vstack((Sr, St))
            S_1p = np.hstack((S_1p, Sr))
        
        return X_t, S_1p, Y_1p

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

    def findDiffusionProcess(self):
        # get Xt and S_1p (from Time Series X)
        Xt, S, Y = self.getArrangedTimeSeriesData()

        # form A
        A = np.hstack((S, Y))

        # hyperparameters
        p = self.p
        beta_1 = self.beta_1

        # dimensions
        L = self.L
        T = self.T
        K = self.K
        
        # estimate the diffusion process (L2 regularized least squares)
        D = np.linalg.inv( np.matmul(A.T, A) + (T-p)*beta_1*np.identity(K*p + L*p) )
        x = np.matmul(D, np.matmul(A.T, Xt))

        # obtain h and c
        h = x[:K*p]
        m = x[K*p:]

        # obtain the diffusion matrices
        H = np.empty((0, L, L))
        for i in range(p):
            Hi = self.unvectorize(h[K*i:K*i+K])
            np.fill_diagonal(Hi, m[L*i:L*i+L])
            Hi = np.expand_dims(Hi, 0)
            H = np.concatenate((H, Hi), axis=0)
        H[abs(H) < 1e-5] = 0
        
        return H

    def getSetDandDcomp(self):
        Uni = np.arange(0, self.L**2, 1)
        D = np.diag(Uni.reshape(self.L, self.L))
        Dcomp = np.delete(Uni, D)
        return D, Dcomp

    def findGraph(self):
        # dimensions
        L = self.L

        # find the diffusion process
        H = self.findDiffusionProcess().squeeze()

        # get the eigendecomposition of the diffusion process
        U, V = np.linalg.eig(H)
        V_hat = np.zeros((L, L))
        for i in range(self.p):
            if np.sign(V[0, 0, 0]) == np.sign(V[i, 0, 0]):
                V_hat += V[i]
            else:
                V_hat -= V[i]
        V_hat /= self.p
        V = V_hat

        # get U
        U = khatri_rao(V, V)
        Uh, S, Vh = np.linalg.svd(U)
        S[-1] = 0
        S[-2] = 0
        S_mod = np.concatenate((np.diag(S), np.zeros((56, 8))))
        U = np.matmul(np.matmul(Uh, S_mod), Vh)
        
        # get set D
        D, Dcomp = self.getSetDandDcomp()

        # test the feasibility of the problem
        rank_U_D = np.linalg.matrix_rank(U[D])
        if rank_U_D <= L-1:
            print("Rank of W_D : {}, Problem is Feasible".format(rank_U_D))
        else: 
            print("Rank of W_D : {}, Problem is infeasible".format(rank_U_D))
        
        # # compute R   
        M_full = np.identity(L**2) - np.matmul(U, np.linalg.pinv(U))
        M = M_full[Dcomp]
        # e1 = np.zeros((L, 1))
        # e1[0] = 1
        # l = np.ones((L-1, 1))
        ll = np.zeros((len(Dcomp), 1))
        ll[np.arange(0, self.L-1, 1)] = 1
        # R = np.hstack((M, np.kron(e1, l)))
        R = np.hstack((M, ll))

        # get b
        b = np.zeros((L**2 + 1, 1))
        b[-1] = 1

        # solve basis pursuit with noisy observations
        beta = 0.001
        A = R.T
        w_Dcomp = np.array(l1regls(matrix(A / np.sqrt(beta)), matrix(b / np.sqrt(beta))))

        # get adjacency matrix
        W = self.unvectorize(w_Dcomp.squeeze())
        
        # w = np.zeros((L*L, ))
        # w[Dcomp] = w_Dcomp.squeeze()
        # W = w.reshape(L, L).T
        W[abs(W) < 1e-5] = 0
        return W

    def findGraphLaplacian(self):
        # dimensions
        L = self.L

        # hyperparams
        p = self.p
        beta_2 = self.beta_2

        # find the diffusion process
        H = self.findDiffusionProcess().squeeze()

        # get the eigendecomposition of the diffusion process
        Up, Vp = np.linalg.eig(H)

        Wp = np.empty((0, L, L))
        for i in range(p):
            # get eignevectors of H_p
            V = Vp[i]

            # get U
            U = khatri_rao(V[:, 1:], V[:, 1:])
            
            # get set D
            D, Dcomp = self.getSetDandDcomp()

            # test the feasibility of the problem
            # rank_U_D = np.linalg.matrix_rank(U[D])
            # if rank_U_D <= L-1:
            #     print("Rank of W_D : {}, Problem is Feasible".format(rank_U_D))
            # else: 
            #     print("Rank of W_D : {}, Problem is infeasible".format(rank_U_D))
            
            # compute A   
            Q_full = np.identity(L**2) - np.matmul(U, np.linalg.pinv(U))
            Q = Q_full[Dcomp]
            A = Q.T

            # compute b
            b = -np.matmul(Q_full[D].T, np.ones((L, 1)))

            # solve basis pursuit with noisy observations
            # l_Dcomp = np.array(l1regls(matrix(A / np.sqrt(beta_2)), matrix(b / np.sqrt(beta_2))))
            l_Dcomp = np.matmul(np.linalg.inv( np.matmul(A.T, A) + beta_2*np.identity(A.shape[1]) ), np.matmul(A.T, b))

            # obtain W from L
            l = np.zeros((L*L, ))
            l[Dcomp] = l_Dcomp.squeeze()
            Laplacian = l.reshape(L, L).T
            np.fill_diagonal(Laplacian, 1)
            Laplacian[abs(Laplacian) < 1e-5] = 0
            W = np.identity(L) - Laplacian
            Wp = np.vstack((Wp, W.reshape(1, L, L)))
        W = np.mean(Wp, axis=0)
        return W



