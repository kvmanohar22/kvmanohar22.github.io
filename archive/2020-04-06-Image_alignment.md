---
layout: post
title: Image Alignment
comments: true
---

<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

Image alignment between two images \\(I_k\\) and \\(I_{k-1}\\) arises in SLAM problems where the objective is to estimate relative rigid body transformation between the above two images.

Let's consider a more simpler problem of sparse image alignment where we use only specific pixels (pixels with enough gradient) in image \\(I_{k-1}\\).

Using a direct approach


$$
\delta I(T_{k,k-1}u_i) = I_k\left( \pi \left(T_{k,k-1} \cdot \rho_i \right) \right) - I_{k-1}\left( \pi \cdot \rho_i \right)
$$

Our objective is to find the following;

$$
T_{k,k-1} = \underset{T_{k,k-1}}{\text{arg min}}\frac{1}{2}\sum_{i \in \mathcal{R}}\lVert\delta I(T_{k,k-1}, u_i)\lVert^2
$$

The above is a non-linear optimization problem which requires us to linearize the problem by introducing a small perturbation around the current estimate.

If we let \\(\xi \in \mathcal{R}^6\\) denote small perturbation on the tangent space then \\(T(\xi) = \exp(\left[\xi\right]_x)\\) is a small perturbation on the manifold \\(SO(3)\\).

Introducing the above perturbation and substituting we have,


$$
\delta I(\xi, u_i) = I_k\left( \pi \left(T_{k,k-1} \cdot \rho_i \right) \right) - I_{k-1}\left( \pi \left( T(\xi) \cdot \rho_i \right) \right)
$$

