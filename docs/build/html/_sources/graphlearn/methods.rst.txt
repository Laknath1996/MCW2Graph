********************************
Graph Topology Learning Methods
********************************

Consider a multi-channel window :math:`\mathbf{X} = [\mathbf{y}_1^\top, \mathbf{y}_2^\top, \dots, \mathbf{y}_L^\top]^\top 
\in \mathbb{R}^{L \times T}`. Here, :math:`L` is the number of channels, :math:`T` is the 
number of time points/ samples, and :math:`\mathbf{y}_i = \big[y_i(t), y_i(1), \dots, y_i(T-1)\big]^\top` 
is the vector including the signal values in channel :math:`i`.

Another way of looking at the multi-channel window considered above is taking it as :math:`\mathbf{X}
= [ \mathbf{x}_0, \mathbf{x}_1, \dots, \mathbf{x}_{T-1}]` where :math:`\mathbf{x}_n = [x_{1,t}, x_{2,t}, \dots, x_{L,t}] \in 
\mathbb{R}^L`. We can consider :math:`\mathbf{x}_n` to be a graph signal at time point :math:`t`, defined on the set of nodes
:math:`\mathcal{V} = \{1, 2, \dots, L\}` of an unknown weighted graph :math:`\mathcal{G}_{\mathbf{X}}(\mathcal{V}, \mathcal{E}, \mathbf{W})` that
underlies :math:`\mathbf{X}`. Here, :math:`\mathcal{E} \subseteq \mathcal{V} \times \mathcal{V}` is the set of edges and 
:math:`W : \mathcal{V} \times \mathcal{V} \rightarrow \mathbb{R}_{+}` is the adjacency matrix. The scalar 
value :math:`x_{i,t}` is the graph signal value at the :math:`i` th node at the :math:`t` th time point.

The :math:`i` th node in :math:`\mathcal{V}` corresponds to the :math:`i` th
channel in :math:`\mathbf{X}`. The edge :math:`(i, j)` in :math:`\mathcal{E}` represents 
the connection between the :math:`i` th and :math:`j` th channels. The entry
:math:`\mathbf{W}_{ij}` represents the strength of the connection between the :math:`i` th and :math:`j` th channels. 

.. currentmodule:: graph_learning.methods

Correlation based Graph Topology Learning
-------------------------------------------

We compute the sample correlation coefficient :math:`r_{ij}` between :math:`i` and :math:`j` channels using
the following equation.

.. math::
    r_{ij} = \frac{(\mathbf{y}_i - \bar{\mathbf{y}_i})^\top (\mathbf{y}_j - \bar{\mathbf{y}_j})}{\Vert \mathbf{y}_i - \bar{\mathbf{y}_i}\Vert_2 \Vert \mathbf{y}_j - \bar{\mathbf{y}_j})\Vert_2}

where, :math:`\bar{y_i} = \big( T^{-1} \sum_{t=0}^{T-1} y_i(t) \big) \mathbf{1}`.

The following hypothesis testing is employed to test whether the population correlation 
coefficient :math:`\rho_{ij}` is significant.

.. math::
    H_0 \; : \; \rho_{i,j} = 0
.. math::
    H_1 \; : \; \rho_{i,j} \neq 0
.. math::
    t^\ast = \frac{r_{ij}\sqrt{T-2}}{\sqrt{1-r_{ij}^2}} \: \sim \: t_{T-2}

Thereafter, we construct :math:`\mathbf{W}` as follows.

.. math::
    \mathbf{W}_{ij} = 
    \begin{cases} 
      |r_{ij}|, & \rho_{ij} \text{ is significant} \\
      0, & \rho_{ij} \text{ is not significant} \\
    \end{cases}

.. autoclass:: CorrelationGraphLearn
    :members: 

**References**

[1] G. B. Giannakis, Y. Shen and G. V. Karanikolas, "Topology Identification and Learning over Graphs: Accounting for Nonlinearities and Dynamics," in Proceedings of the IEEE, vol. 106, no. 5, pp. 787-807, May 2018, doi: 10.1109/JPROC.2018.2804318.
        
[2] E. D. Kolaczyk, Statistical Analysis of Network Data: Methods and Models. New York, NY, USA: Springer-Verlag, 2009.

Partial Correlation based Graph Topology Learning
---------------------------------------------------

Here, we assume that each :math:`\mathbf{x}_t` is a Gaussian random vector. Then we can compute the sample partial correlation coefficient :math:`\tilde{r}_{ij}` between :math:`i` 
and :math:`j` channels using the following equation.

.. math::
    \tilde{r}_{ij} = - \frac{\hat{\Theta}_{ij}}{\sqrt{\hat{\Theta}_{ii} \hat{\Theta}_{jj}}}

where, :math:`\hat{\Theta}^{-1} = \hat{\Sigma} = T^{-1} (\mathbf{X} - \mathbf{1}\bar{\mathbf{x}}^\top)(\mathbf{X} - \mathbf{1}\bar{\mathbf{x}}^\top)^\top` 
is the sample covariance matrix.

The following hypothesis testing is employed to test whether the population correlation 
coefficient :math:`\tilde{\rho}_{ij}` is significant.

.. math::
    H_0 \; : \; \tilde{\rho}_{i,j} = 0
.. math::
    H_1 \; : \; \tilde{\rho}_{i,j} \neq 0
.. math::
    t^\ast = \tilde{r}_{ij} \sqrt{\frac{T - L}{1 - \tilde{r}_{ij}^2}} \: \sim \: t_{T-L}

Thereafter, we construct :math:`\mathbf{W}` as follows.

.. math::
    \mathbf{W}_{ij} = 
    \begin{cases} 
      |\tilde{r}_{ij}|, & \tilde{\rho}_{ij} \text{ is significant} \\
      0, & \tilde{\rho}_{ij} \text{ is not significant} \\
    \end{cases}

.. autoclass:: PartialCorrelationGraphLearn
    :members:

**References**

[1] G. B. Giannakis, Y. Shen and G. V. Karanikolas, "Topology Identification and Learning over Graphs: Accounting for Nonlinearities and Dynamics," in Proceedings of the IEEE, vol. 106, no. 5, pp. 787-807, May 2018, doi: 10.1109/JPROC.2018.2804318.
        
[2] E. D. Kolaczyk, Statistical Analysis of Network Data: Methods and Models. New York, NY, USA: Springer-Verlag, 2009.

Sparse Inverse Covariance (Graphical Lasso) based Graph Topology Learning
--------------------------------------------------------------------------

We solve the following optimization problem to estimate a sparse inverse covariance
matrix :math:`\hat{\Theta}`.

.. math::
    \Theta^\ast = \underset{\Theta \succcurlyeq 0}{\mathrm{argmin }} \: -\log (\det \Theta) + \text{tr} (\hat{\Sigma} \Theta) + \beta \Vert \Theta \Vert_1

where, :math:`\hat{\Theta}^{-1} = \hat{\Sigma} = T^{-1} (\mathbf{X} - \mathbf{1}\bar{\mathbf{x}}^\top)(\mathbf{X} - \mathbf{1}\bar{\mathbf{x}}^\top)^\top` 
is the sample covariance matrix. The higher the :math:`\beta`, the higher the sparsity of the resulting graph topology.

This optimization is solved using the ADMM described in [3]. The following are the iteration steps.

.. math::
    \begin{align}
        \Theta_{k+1} &= \mathcal{F}_{\gamma} (\psi_k - \frac{1}{\gamma} \Lambda_k - \frac{1}{\gamma} \hat{\Sigma}) \\
        \psi_{k+1} &= \text{ST}_{\beta / \gamma} (\Theta_{k+1} - \frac{1}{\gamma} \Lambda_k) \\
        \Lambda_{k+1} &= \Lambda_k + \gamma (\Theta_{k+1} - \psi_{k+1})
    \end{align}

where, :math:`\mathcal{F}_{\gamma} (\mathbf{X} = \mathbf{U} \text{diag} \{ \lambda_1, \dots, \lambda_n \} \mathbf{U}^\top) = \frac{1}{2} \mathbf{U} \text{diag} \big\{ \lambda_i + \sqrt{\lambda_i^2 + 4 / \gamma} \big\} \mathbf{U}^\top`, and 
:math:`\text{ST}` is the soft-thresholding operator. 

After solving, we compute the partial
correlation coefficients.

.. math::
    \tilde{r}_{ij} = - \frac{\Theta^{\ast}_{ij}}{\sqrt{\Theta^{\ast}_{ii} \Theta^{\ast}_{jj}}}

Thereafter, we test for the statistical significance of the partial correlation coefficients using 
the following test.

.. math::
    H_0 \; : \; \tilde{\rho}_{i,j} = 0
.. math::
    H_1 \; : \; \tilde{\rho}_{i,j} \neq 0
.. math::
    t^\ast = \tilde{r}_{ij} \sqrt{\frac{T - L}{1 - \tilde{r}_{ij}^2}} \: \sim \: t_{T-L}

Finally, we construct :math:`\mathbf{W}` as follows.

.. math::
    \mathbf{W}_{ij} = 
    \begin{cases} 
      |\tilde{r}_{ij}|, & \tilde{\rho}_{ij} \text{ is significant} \\
      0, & \tilde{\rho}_{ij} \text{ is not significant} \\
    \end{cases}

.. autoclass:: GraphicalLassoGraphLearn
    :members:

**References**
        
[1] J. Friedman, T. Hastie, and R. Tibshirani, “Sparse inverse covariance estimation with the graphical lasso,” Biostatistics, vol. 9,
no. 3, pp. 432–441, Jul. 2008.

[2] G. B. Giannakis, Y. Shen and G. V. Karanikolas, "Topology Identification and Learning over Graphs: Accounting for Nonlinearities and Dynamics," in Proceedings of the IEEE, vol. 106, no. 5, pp. 787-807, May 2018, doi: 10.1109/JPROC.2018.2804318.

[3] Yuxin Chen, "Alternating direction method of multipliers", ELE 522: Large-Scale Optimization for Data Science, Princeton University, Fall 2019.


Smoothness based Graph Topology Learning
------------------------------------------

The following optimization problem is solved to obtain the adjacency matrix of the underlying 
graph topology.

.. math::
    \mathbf{W}^\ast = \underset{\mathbf{W} \in \mathcal{W}}{\mathrm{argmin }} \: \Vert \mathbf{W} \circ \mathbf{Z} \Vert_1 - \alpha \mathbf{1}^\top \log \mathbf{W}\mathbf{1} + \beta \Vert \mathbf{W} \Vert_F^2

where, :math:`\mathcal{W} = \{ \mathbf{W} | \mathbf{W}_{ij} = \mathbf{W}_{ji} \geq 0, \text{diag} \{ \mathbf{W} \} = \mathbf{0} \}` is the set of valid adjacency matrices and 
:math:`\mathbf{Z}_{ij} = \Vert \mathbf{y}_i - \mathbf{y}_j \Vert_2^2` is the pairwise distance matrix. Bigger the :math:`\alpha`, the bigger the weights of :math:`\mathbf{W}` and bigger the :math:'\beta', the denser the graph. Vectorizing this expression, we get, 

.. math::
    \mathbf{w}^\ast = \underset{\mathbf{w} \in \mathbb{R}_+^K}{\mathrm{argmin }} \: \mathbb{1} \{ \mathbf{w} \geq \mathbf{0} \} + 2\mathbf{w}^\top \mathbf{z} - \alpha \mathbf{1}^\top \log \mathbf{d} + \beta \Vert \mathbf{w} \Vert_2^2

where, :math:`\mathbf{w}` and :math:`\mathbf{z}` contains the upper triangular (w.o the primary diagonal) elements of :math:`\mathbf{W}` and :math:`\mathbf{Z}` (therefore, :math:`K = L(L-1)/2`).
:math:`\mathbf{d}` is the degree vector of :math:`\mathbf{W}` such that :math:`\mathbf{W} \mathbf{1} = \mathbf{S} \mathbf{w} = \mathbf{d}`, where :math:`\mathbf{S}` is a fixed linear operator. 

The iterative steps of this optimization are clearly stated in [1].

**References**
        
[1] Vassilis Kalofolias, "How to Learn a Graph from Smooth Signals", Proceedings of the 19th International Conference on Artificial Intelligence and Statistics, PMLR 51:920-929, 2016.


