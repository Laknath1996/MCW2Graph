********************************
Graph Topology Learning Methods
********************************

Consider a multi-channel window :math:`\mathbf{X} = [y_0^\top, y_1^\top, \dots, y_{L-1}^\top]^\top 
\in \mathbb{R}^{L \times N}`.Here, :math:`L` is the number of channels, :math:`N` is the 
number of time points/ samples, and :math:`y_i = \big[y_i[0], y_i[1], \dots, y_i[N-1]\big]^\top` 
is the vector including the signal values in channel :math:`i`.

Let :math:`\mathcal{G} = (\mathcal{V}, \mathcal{E})` be the underlying graph of the 
muli-channel window :math:`\mathbf{X}`, characterized by the adjancency matrix :math:`\mathbf{W}`.

.. currentmodule:: graph_learning.methods

Correlation based Graph Topology Learning
-------------------------------------------

We compute the sample correlation coefficient :math:`r_{ij}` between :math:`i` and :math:`j` channels using
the following equation.

.. math::
    r_{ij} = \frac{(y_i - \bar{y_i})^\top (y_j - \bar{y_j})}{\Vert y_i - \bar{y_i}\Vert_2 \Vert y_j - \bar{y_j})\Vert_2}

where, :math:`\bar{y_i} = \big( N^{-1} \sum_{n=0}^{N-1} y_i[n] \big) \mathbf{1}`.

The following hypothesis testing is employed to test whether the population correlation 
coefficient :math:`\rho_{ij}` is significant.

.. math::
    H_0 \; : \; \rho_{i,j} = 0
.. math::
    H_1 \; : \; \rho_{i,j} \neq 0
.. math::
    t^\ast = \frac{r_{ij}\sqrt{n-2}}{\sqrt{1-r_{ij}^2}} \: \sim \: t_{n-2}

Thereafter, we construct :math:`\mathbf{W}` as follows.

.. math::
    \mathbf{W}_{ij} = 
    \begin{cases} 
      |r_{ij}|, & \rho_{ij} \text{ is significant} \\
      0, & \rho_{ij} \text{ is not significant} \\
    \end{cases}

.. autoclass:: CorrelationGraphLearn
    :members: 

Partial Correlation based Graph Topology Learning
---------------------------------------------------

We compute the sample partial correlation coefficient :math:`\tilde{r}_{ij}` between :math:`i` 
and :math:`j` channels using the following equation.

.. math::
    \tilde{r}_{ij} = \frac{\Theta_{ij}}{\sqrt{\Theta_{ii} \Theta{jj}}}

where, :math:`\Theta = \text{cov}(\mathbf{X})`.

The following hypothesis testing is employed to test whether the population correlation 
coefficient :math:`\tilde{\rho}_{ij}` is significant.

.. math::
    H_0 \; : \; \tilde{\rho}_{i,j} = 0
.. math::
    H_1 \; : \; \tilde{\rho}_{i,j} \neq 0
.. math::
    t^\ast = \tilde{r}_{ij} \sqrt{\frac{n - L}{1 - \tilde{r}_{ij}^2}} \: \sim \: t_{n-L}

Thereafter, we construct :math:`\mathbf{W}` as follows.

.. math::
    \mathbf{W}_{ij} = 
    \begin{cases} 
      |\tilde{r}_{ij}|, & \tilde{\rho}_{ij} \text{ is significant} \\
      0, & \tilde{\rho}_{ij} \text{ is not significant} \\
    \end{cases}


.. autoclass:: PartialCorrelationGraphLearn
    :members: