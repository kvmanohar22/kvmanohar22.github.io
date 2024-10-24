---
layout: post
title: Optimal Controls 3
comments: true
---

<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

Non-holonomic integrator is given by the following dynamics;

$$
\begin{align}
\dot x_1 &= u_1 \\
\dot x_2 &= u_2 \\
\dot x_3 &= x_1 u_2 - x_2 u_1
\end{align}
$$

Consider the objective of driving the system from \\((0, 0, 0)^T\\) to \\((0, 0, a)^T\\) by minimizing the following objective (minimum energy).

$$
J({\bf u}) = \int_{t=0}^{t=1} {\bf u}^T(t) {\bf u}(t) dt
$$

Turns out that for such a state transfer, optimal inputs are sinusoidal with fundamental frequency \\(2\pi\\).
