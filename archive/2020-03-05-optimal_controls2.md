---
layout: post
title: Optimal Controls 2
comments: true
---

<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

Contents:
- [Problem statement](#problem)
- [Fixed time and Fixed end point](#ftfe)
  - [Solution using Calculus of Variations](#cv)
  - [Solution using Dynamic programming](#dp)
- [Fixed time and Free end point](#ftffe)
  - [Solution using Calculus of Variations](#cv)
  - [Solution using Dynamic programming](#dp)
- [Free time and Fixed end point](#fftfe)
  - [Solution using Calculus of Variations](#cv)
  - [Solution using Dynamic programming](#dp)
- [End point lying on a manifold](#manifold)
- [Moving end point](#moving)
- [End point moving on a manifold](#manifold_moving)

This post discusses different solutions obtained (open loop v/s closed loop) by applying different theories (calculus of variations, dynamic programming) for solving optimal control problems of a simple linear dynamical system.


<a name='problem'/>
## Problem statement

Consider the minimum energy problem of a linear system (later we will see this example for a double integrator and harmonic oscillator).

$$
J(u) = \frac{1}{2}\int_{t_0}^{t_f} u^T(t) u(t) dt \tag{1}
$$

subject to:

$$
\begin{align}
  \dot {\bf x}(t) &= A{\bf x}(t) + Bu(t) \\
       {\bf x}(t_0) &= x_0 \\
       {\bf x}(t_f) &= {\bf x_f} \tag{2}
\end{align}
$$

Now, let's look at deriving the solution to the above problem for various initial and final conditions using two different approaches for each type.

Let's start with the simplest case.

<a name='ftfe'/>
## 1. Fixed time and Fixed end point
In this problem, \\(\delta t = 0\\) and \\(\delta {\bf x}_f = 0\\)
<a name='cv'/>
### Calculus of Variations solution
From the previous post, using Hamiltonian, we have:

$$
H := p^T(t)f(t, x, u) - \mathcal{L} = p^T(t)(Ax+Bu)-\frac{1}{2}u^Tu
$$

From \\(H_x = -\dot p(t)\\), we get;

$$
\dot p(t) = -A^Tp(t)
$$

and hence;

$$
p(t) = e^{A^T(t_f - t)}p(t_f)
$$

From \\(H_u = 0\\), we get;

$$
u^*(t) = B^Tp(t)
$$

and hence;

$$
x^*(t) = Ax + BB^Tp(t)
$$


Solving the system dynamics, gives;

$$
\begin{align}
x(t) &= e^{A(t-t_0)}x(t_0) + \int_{t_0}^{t_f} e^{A(t-\tau)}Bu(\tau) d\tau \\
     &= e^{A(t-t_0)}x(t_0) + {\Big [}\int_{t_0}^{t_f} e^{A(t-\tau)}BB^Te^{A(t-\tau)} d\tau \Big{]} p(t_f)
\end{align}
$$

Using the boundary condition \\(x(t_f) = x_f \\), we can obtain the expression for p(t_f) as follows;

$$
\begin{align}
x(t_f) &= e^{A(t_f-t_0)}x(t_0) + {\Big [}\int_{t_0}^{t_f} e^{A(t_f-\tau)}BB^Te^{A(t_f-\tau)} d\tau \Big{]} p(t_f) \\
       &= e^{A(t_f-t_0)}x(t_0) + W_{t_{f}}(A, B)p(t_f)
\end{align}
$$

where \\(W_{t_{f}}\\) is Controllability Gramian. Assuming that the system is controllable (and hence \\(W_{t_{f}}\\) is invertible),

$$
\begin{align}
p(t_f) &= W_{t_{f}}(A, B)^{-1}\left(x(t_f) - e^{A(t_f-t_0)}x(t_0)\right)
\end{align}
$$

Therefore, the optimal inputs is given by;

$$
\begin{align}
  u^*(t) &= B^Tp(t) \\
         &= B^Te^{A^T(t_f-t)} W_{t_{f}}(A, B)^{-1}x(t_f)
\end{align}
$$

As you can see, this is an open-loop solution which is not practical in deploying into real world.

<a name='dp'/>
### Dynamic programming solution

From the previous post, we have;

$$
-\frac{\partial V(x, t)}{\partial t} = \underset{u}{\text{Inf}}\Big\{\mathcal{L} + <\frac{\partial V(x, t)}{\partial x}, f(t, x)>\Big\}
$$

Note that we cannot blindly apply the above HJB equation to the problem stated earlier. The reason will become quite clear shortly. Let's consider a slightly modified problem as follows;

$$
J(u) = \frac{1}{2}({\bf x_f} - {\bf x}(t_f))^TQ_f({\bf x_f} - {\bf x}(t_f)) + \frac{1}{2}\int_{0}^{T} u^T(t) u(t) dt
$$

subject to:

$$
\begin{align}
  \dot {\bf x}(t) &= A{\bf x}(t) + Bu(t) \\
       {\bf x}(t_0) &= x_0 \\
       {\bf x}(t_f) &= x_f
\end{align}
$$

where we have added in addition a terminal state cost.




<a name='ftffe'/>
## 2. Fixed time and Free end point

Intuitively thinking when do we encounter a situation of free end point type? Say, we start from some point \\(x(t_0) = x_0\\) and we want to reach \\(x(t_f) = 0 \\). This is a very hard constraint that which would not be accomplished exactly. But for the practical applications, if we are very close to origin, then the task would be complete. Instead of defining a terminal hard constraint as \\(x(t_f) = 0\\), we could add some high penalty for not being at origin in our cost. This is a soft constraint and is formulated as follows.

Let's modify the objective function by adding terminal state cost so that HJB equations can be solved without shift of origin;
$$
J(u) = \frac{1}{2}{\bf x}^T_fQ_f{\bf x}_f + \frac{1}{2}\int_{0}^{T} u^T(t) u(t) dt
$$

subject to:

$$
\begin{align}
  \dot {\bf x}(t) &= A{\bf x}(t) + Bu(t) \\
       {\bf x}(t_0) &= x_0 \\
\end{align}
$$

where \\(Q_f\\) is a symmetric positive-definite matrix and we want to reach origin at \\(t_f\\).
In this problem, \\(\delta t = 0\\) and \\(\delta {\bf x}_f \neq 0\\)
<a name='cv'/>
### Calculus of Variations solution
<a name='dp'/>
### Dynamic programming solution
<a name='fftfe'/>
## 3. Free time and Fixed end point
In this problem, \\(\delta t \neq 0\\) and \\(\delta {\bf x}_f = 0\\)
<a name='cv'/>
### Calculus of Variations solution
<a name='dp'/>
### Dynamic programming solution
<a name='manifold'/>
## 4. End point lying on a manifold
In this problem, \\(x(t_f) \in S\\) where \\(S(x_1, \dots x_n) = 0\\) is a manifold.
<a name='moving'/>
## 5. Moving end point
In this problem, \\(x(t_f) = \gamma (t_f)\\) where \\(\gamma(t)\\) is a parameterized curve.
<a name='manifold_moving'/>
## 6. End point moving on a manifold
