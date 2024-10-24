---
layout: post
title: Structureless optimization
comments: true
---

<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      inlineMath: [ ['$','$'], ["\\(","\\)"] ],
      processEscapes: true
    }
  });
</script>
    
<script type="text/javascript"
        src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>

In this post, I will be going over Bundle Adjustment. Content in this post is summarization of material from the paper [1].

Contents are as follows:

- [Problem Statement](#ps)
- [Background](#background)
  - [SVD](#svd)
  - [Orthogonal Projections](#ortho)
- [Derivation](#solution)
- [References](#ref)

<a name="ps"/>
## Problem Statement
Suppose we have \\(L\\) landmarks and \\(K\\) frames such that the landmark \\(l\\) is observed by subset of frames \\(X(l)\\). The objective function that we try to minimize is thus:

$$
J = \sum_{l=1}^{L}\sum_{i \in X(l)} || \mathbf{r}_{\mathcal{C}_{il}}||_{\mathbf{\Sigma}_c}^2
$$

where $X(l)$ is subset of frames that observe the landmark \\(l\\), \\(\Sigma_c\\) is measurement covariance of the pixel and $$\mathbf{r}_{\mathcal{C}_{il}}$$ is the standard reprojection error given by;

$$
\mathbf{r}_{\mathcal{C}_{il}}(\mathtt{R}_i, \mathbf{p}_i, \boldsymbol{\rho}_l) = \mathbf{z}_{il} - \pi\left( \mathtt{R}_i \boldsymbol{\rho}_l + \mathbf{p}_i \right)
$$

Since $$\mathbf{r}_{\mathcal{C}_{il}}$$ is a nonlinear function, we solve the above iteratively using Gauss-Newton method or Levenberg-Marquardt method. Derivation of the method is presented in [Section Derivation](#solution).

<a name="background"/>
## Background


<a name="background"/>
### SVD
We shall be using SVD to get orthonormal basis for a linear transformation. Let \\(A \in \mathbb{R}^{n \times m}\\). Since we will be mostly interested in over determined system of equations, we have \\(n \gt m\\). Let us further assume \\(A\\) is a full column rank matrix.

$$
A = U \Sigma V^T
$$
where \\(U \in \mathbb{R}^{n \times n}, \Sigma \in \mathbb{R}^{n \times m}, V \in \mathbb{R}^{m \times m}\\). 

\begin{align}
AV &= U \Sigma \\
\end{align}

$$
Av_i  = \begin{cases}
        \sigma_i u_i & \text{if 1 $\leq$ i $\leq$ m} \\
        0 & \text{otherwise}
      \end{cases}
$$

<a name="ortho"/>
### Orthogonal Projections
Let \\(V = \mathbb{R}^{n}\\) be an inner product space. Let \\(W\\) be a subspace of \\(V\\). Let \\(\\{w_1, w_2, \dots, w_k\\}\\) be a basis for \\(W\\). Let \\(x \in V\\) and \\(y \in W\\) be it's orthogonal projection i.e, \\(y = Ex\\) where \\(E\\) is orthogonal projection operator from \\(V\\) to \\(W\\) along \\(W^\bot\\). This is illustrated in the following figure.

Let us find structure of the operator \\(E\\). Since \\(y \in W\\), \\(y = \alpha_1 w_1 + \dots + \alpha_k w_k\\) or \\(y = A{\bf \alpha}\\) where \\(\alpha \in \mathbb{R}^k\\) and vectors \\(w_i\\) are stacked onto columns of \\(A\\) and hence \\(A \in \mathbb{R}^{n \times k}\\). Further \\(y - x \in W^\bot\\). So \\((y-x)^\mathsf{T}w_i = 0 \enspace \forall \enspace i = 1, 2, \dots, k\\). or \\(A^\mathsf{T}(y-x) = 0\\). Expanding the above, we get; \\(A^\mathsf{T}y = A^\mathsf{T}x\\) and since \\(y = A{\bf \alpha}\\), we have \\(A^\mathsf{T}A{\bf \alpha} = A^\mathsf{T}x\\) or \\({\bf \alpha} = (A^\mathsf{T}A)^{-1}A^\mathsf{T}x \\) and hence \\(y = A(A^\mathsf{T}A)^{-1}A^\mathsf{T}x\\). Hence by simple inspection we have;

$$
E := A(A^\mathsf{T}A)^{-1}A^\mathsf{T} \label{1}
$$

Note that inverse of \\(A^\mathsf{T}A\\) is well defined since \\(A\\) has full column rank. Clearly range space of \\(E\\) is \\(\mathcal{R}(E) = W\\). Let \\(x \in \mathcal{N}(E) \Leftrightarrow Ex = 0 \Leftrightarrow x - 0 \in W^\bot \Leftrightarrow x \in W^\bot\\). And hence \\(\mathcal{N}(E) = W^\bot\\).

If we define \\(F := I - E = I - A(A^\mathsf{T}A)^{-1}A^\mathsf{T}\\), then \\(F\\) is again an orthogonal projection operator but from \\(V\\) to \\(W^\bot\\). This is because if \\(x \in V\\) then \\(Fx = x - Ex = x - y\\) where \\(y = Ex\\). As noted earlier, \\(y-x \in W^\bot\\). And hence \\(F\\) is an orthogonal projection operator from \\(V\\) to \\(W^\bot\\).

In summary, what we have seen is: if columns of matrix \\(A\\) span a subspace \\(W\\) of an inner product space \\(V\\), then \\(E := A(A^\mathsf{T}A)^{-1}A^\mathsf{T}\\) is an orthogonal projection operator from \\(V\\) to \\(W\\). Further \\(F := I - E\\) is an orthogonal projection operator from \\(V\\) to \\(W^\bot\\).

Further, if \\(B\\) is a basis for \\(W^\bot\\), then we can define \\(\tilde{F} := B(B^\mathsf{T}B)^{-1}B^\mathsf{T}\\) to be orthogonal projection operator from \\(V\\) to \\(W^\bot\\).

We shall be using this in modifying our objective function (1) to make the computation more efficient.

<a name="solution"/>
## Derivation

$$
J = \sum_{l=1}^{L}\sum_{i \in X(l)} || \mathbf{r}_{\mathcal{C}_{il}}||_{\Sigma_c}^2
$$

Since the above cost function is non-linear, solution is obtained iteratively solving for incremental updates by linearizing around current estimate.

If \\( (\delta \boldsymbol{\phi}_i \ \ \delta \mathbf{p}_i \ \ \delta \boldsymbol{\rho}_l) \\) are incremental updates for \\(\mathtt{R}_i \ \, \mathbf{p}_i \ \, \boldsymbol{\rho}_l \\) respectively, then updates are given by:

$$
\begin{align}
  \mathtt{R}_i &\leftarrow \mathtt{R}_i \mathrm{Exp}(\delta \boldsymbol{\phi}_i) \\
  \mathbf{p}_i &\leftarrow \mathbf{p}_i + \mathtt{R}_i \delta \mathbf{p}_i \\
  \boldsymbol{\rho}_l &\leftarrow \boldsymbol{\rho}_l + \delta\boldsymbol{\rho}_l 
\end{align}
$$

The jacobians of the above cost function can now be calculated by expanding the cost function using Taylor series truncated upto first order,

$$
\begin{align}
  \mathbf{r}_{\mathcal{C}_{il}}(\boldsymbol{\rho}_l+\delta\boldsymbol{\rho}_l)
    &= \mathbf{z}_{il} - \pi \left( \mathtt{R}_i(\boldsymbol{\rho}_l+\delta\boldsymbol{\rho}_l)+\mathbf{p}_i\right) \\
    &= \mathbf{z}_{il} - \pi \left( \mathtt{R}_i\boldsymbol{\rho}_l+\mathbf{p}_i + \mathtt{R}_i\delta\boldsymbol{\rho}_l\right) \\
    &\approx \mathbf{z}_{il} - \pi \left( \mathtt{R}_i\boldsymbol{\rho}_l+\mathbf{p}_i\right) 
        - \frac{\partial \pi(a)}{\partial a}\bigg|_{a = _i\boldsymbol{\rho}_l} \mathtt{R}_i\delta\boldsymbol{\rho}_l \\
    &= \mathbf{r}_{\mathcal{C}_{il}}(\boldsymbol{\rho}_l)
        - \frac{\partial \pi(a)}{\partial a}\bigg|_{a = _i\boldsymbol{\rho}_l} \mathtt{R}_i\delta\boldsymbol{\rho}_l
\end{align}
$$

$$
\begin{align}
  \mathbf{r}_{\mathcal{C}_{il}}(\mathtt{R}_i \mathrm{Exp}(\delta \boldsymbol{\phi}_i))
    &= \mathbf{z}_{il} - \pi \left( \mathtt{R}_i\mathrm{Exp}(\delta \boldsymbol{\phi}_i)\boldsymbol{\rho}_l + \mathbf{p}_i\right) \\
    &= \mathbf{z}_{il} - \pi \left( \mathtt{R}_i (\mathbf{I}_{3\times 3} + \delta \boldsymbol{\phi}_i ^\wedge) \boldsymbol{\rho}_l + \mathbf{p}_i\right) \\
    &= \mathbf{z}_{il} - \pi \left( \mathtt{R}_i \boldsymbol{\rho}_l + \mathbf{p}_i - \mathtt{R}_i \boldsymbol{\rho}_l^\wedge \delta \boldsymbol{\phi}_i \right) \\
    &\approx \mathbf{z}_{il} - \pi \left( \mathtt{R}_i\boldsymbol{\rho}_l+\mathbf{p}_i\right) + \frac{\partial \pi(a)}{\partial a}\bigg|_{a = _i\boldsymbol{\rho}_l} \mathtt{R}_i\boldsymbol{\rho}_l^\wedge\delta\boldsymbol{\phi}_i \\
    &= \mathbf{r}_{\mathcal{C}_{il}}(\mathtt{R}_i) + \frac{\partial \pi(a)}{\partial a}\bigg|_{a = _i\boldsymbol{\rho}_l} \mathtt{R}_i\boldsymbol{\rho}_l^\wedge\delta\boldsymbol{\phi}_i
\end{align}
$$

$$
\begin{align}
  \mathbf{r}_{\mathcal{C}_{il}}(\mathbf{p}_i+\mathtt{R}_i\delta \mathbf{p}_i)
    &= \mathbf{z}_{il} - \pi \left( \mathtt{R}_i \boldsymbol{\rho}_l+\mathbf{p}_i+\mathtt{R}_i\delta \mathbf{p}_i\right) \\
    &\approx \mathbf{z}_{il} - \pi \left( \mathtt{R}_i\boldsymbol{\rho}_l+\mathbf{p}_i\right) 
        - \frac{\partial \pi(a)}{\partial a}\bigg|_{a = _i\boldsymbol{\rho}_l} \mathtt{R}_i\delta \mathbf{p}_i \\
    &= \mathbf{r}_{\mathcal{C}_{il}}(\mathbf{p}_i) - \frac{\partial \pi(a)}{\partial a}\bigg|_{a = _i\boldsymbol{\rho}_l} \mathtt{R}_i\delta \mathbf{p}_i
\end{align}
$$


where $$ _i\boldsymbol{\rho}_l := \mathtt{R}_i \boldsymbol{\rho}_l+\mathbf{p}_i$$. Jacobians can be easily read from the above:

$$
\begin{align}
\frac{\partial {\bf r}_{\mathcal{C}_{il}}}{\partial \boldsymbol{\rho}_l} &= -\frac{\partial \pi(a)}{\partial a} \bigg|_{a = _i\boldsymbol{\rho}_l} \mathtt{R}_i \\
\frac{\partial {\bf r}_{\mathcal{C}_{il}}}{\partial \delta \boldsymbol{\phi}_i} &= \frac{\partial \pi(a)}{\partial a} \bigg|_{a = _i\boldsymbol{\rho}_l} \mathtt{R}_i \boldsymbol{\rho}_l^\wedge \\
\frac{\partial {\bf r}_{\mathcal{C}_{il}}}{\partial \mathbf{p}_i}    &= -\frac{\partial \pi(a)}{\partial a} \bigg|_{a = _i\boldsymbol{\rho}_l} \mathtt{R}_i \\
\end{align}
$$

To make the notation a bit more clear, denote $$\delta \mathbf{T}_{il} := [\delta \boldsymbol{\phi}_i^\mathsf{T}, \delta p_i^\mathsf{T}]^\mathsf{T}$$ and  $$\mathbf{A}_{il} := \frac{\partial {\bf r}_{\mathcal{C}_{il}}}{\partial \delta \boldsymbol{\rho}_l}$$ and 

$$
\mathbf{B}_{il} := \begin{bmatrix}
              \frac{\partial {\bf r}_{\mathcal{C}_{il}}}{\partial \delta \boldsymbol{\phi}_i} & \frac{\partial {\bf r}_{\mathcal{C}_{il}}}{\partial \delta \mathbf{p}_i}
          \end{bmatrix}
$$

Cost function now becomes ($$\mathbf{b}_{il} := -\mathbf{r}_{\mathcal{C}_{il}}(\mathtt{R}_i, \mathbf{p}_i, \boldsymbol{\rho}_l)$$):

$$
J' = \sum_{l=1}^{L}\sum_{i \in X(l)} || \mathbf{B}_{il} \delta \mathbf{T}_{il} + \mathbf{A}_{il} \delta \boldsymbol{\rho}_l - \mathbf{b}_{il}||^2
$$

Let's calculate the dimensions of each of the quantities in above equation. $$\mathbf{B}_{il} \in \mathbb{R}^{2 \times 6}, \enspace \delta \mathbf{T}_{il} \in \mathbb{R}^{6 \times 1}, \enspace \mathbf{A}_{il} \in \mathbb{R}^{2 \times 3}, \enspace \delta \boldsymbol{\rho}_l \in \mathbb{R}^{3 \times 1}, \enspace \mathbf{b}_{il} \in \mathbb{R}^{2 \times 1} $$,   Getting rid of inner summation and stacking the Jacobians, we have:

$$
J' = \sum_{l=1}^{L} || \mathbf{B}_{l} \delta \mathbf{T}_{l} + \mathbf{A}_{l} \delta \boldsymbol{\rho}_l - \mathbf{b}_l||^2
$$

Say a landmark $$l$$ is observed in $$n_l$$ frames, in which case the dimensions of above quantities become: $$\mathbf{B}_l \in \mathbb{R}^{2n_l \times 6n_l}, \enspace \delta \mathbf{T}_{l} \in \mathbb{R}^{6n_l \times 1}, \enspace \mathbf{A}_l \in \mathbb{R}^{2n_l \times 3}, \enspace \delta \boldsymbol{\rho}_l \in \mathbb{R}^{3 \times 1}, \enspace \mathbf{b}_l \in \mathbb{R}^{2n_l \times 1}$$.

The optimal perturbation in landmark $$\delta \boldsymbol{\rho}_l$$ that minimizes the above cost function is: $$\delta \boldsymbol{\rho}_l^\ast = -(\mathbf{A}_l^\mathsf{T}\mathbf{A}_l)^{-1}\mathbf{A}_l^\mathsf{T}(\mathbf{B}_l \delta \mathbf{T}_l - \mathbf{b}_l)$$. Plugging this back into the above equation, we get:

$$
\begin{align}
J' &= \sum_{l=1}^{L} || \mathbf{B}_{l} \delta \mathbf{T}_{l} - \mathbf{A}_{l}(\mathbf{A}_l^\mathsf{T}\mathbf{A}_l)^{-1}\mathbf{A}_l^\mathsf{T}(\mathbf{B}_l \delta T_l - \mathbf{b}_l) - \mathbf{b}_l||^2 \\
   &= \sum_{l=1}^{L} || (\mathbf{I} - \mathbf{A}_{l}(\mathbf{A}_l^\mathsf{T}\mathbf{A}_l)^{-1}\mathbf{A}_l^\mathsf{T})(\mathbf{B}_l \delta \mathbf{T}_l - \mathbf{b}_l)||^2 \\
\end{align}
$$

Denote, $$\mathbf{E}_l := \mathbf{A}_{l}(\mathbf{A}_l^\mathsf{T}\mathbf{A}_l)^{-1}\mathbf{A}_l^\mathsf{T}$$ and $$\mathbf{F}_l := \mathbf{I} - \mathbf{E}_l$$. Clearly from [Orthogonal Projection](#ortho) section, there is quite a similarity here!


The above linear system can now be solved for $$ \delta \mathbf{T}_l $$.


This manipulation of cost function has essentially reduced the number of variables involving both poses and landmarks to only poses. The optimal perturbation for landmarks can be obtained via backsubstitution.

<a name="ref"/>
## References

[1] On-Manifold Preintegration Theory for Fast and Accurate Visual-Inertial Navigation.