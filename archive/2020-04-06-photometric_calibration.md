---
layout: post
title: Photometric calibration
comments: true
---

<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>


**Geometric calibration**: This corresponds to mapping a scene point \\({\bf x} \in \mathbb{R}^3\\) onto the image plane \\({\bf u} \in \Omega\\). Most widely used being pinhole camera model

**Photometric calibration**: This calibration takes into account the actual image formation pipeline at the image plane. i.e, this calibration mathematically formulates right from irradiance recieved at a photon receptor on the camera pixel sensor to the corresponding observed pixel intensity.

If \\({B}\\) denotes irradiance, \\(t\\) denotes exposure time of shutter, \\({V}\\) denotes lens attenuation (vignetting) then we have and \\(G\\) denotes non-linear response function:

$$
I({\bf x}) = G(t {V}({\bf x}) {B}({\bf x}))
$$

where \\(I({\bf x})\\) is the pixel intensity value at pixel location \\({\bf x} \in \Omega\\). Denote \\(I'({\bf x}) := t {B}({\bf x}) = \frac{G^{-1}(I({\bf x}))}{V({\bf x})}\\). Using naming convention as in [1], \\(I'\\) is the photometrically corrected image. Also, note that inverse of \\(G\\) is well defined since, \\(G\\) is a monotonically increasing function.