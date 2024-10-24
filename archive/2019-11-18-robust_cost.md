---
layout: post
head: robust_cost
title: Robust Cost
comments: true
---
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

In robotics, we often come across estimating maximum likelihood or maximum-a-posterior problems quite frequently. And the choice of white noise that we introduce to estimate makes a lot of difference. This blog is about analyzing different cost functions and benchmarking them on some standard problems.

This post is based on the paper titled **At All Costs: A Comparison of Robust Cost Functions for Camera Correspondence Outliers** [1]. This blog is accompanied by code written in C++14 at: [kvmanohar22/robust_cost](https://github.com/kvmanohar22/utils). 
