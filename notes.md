---
layout: default
title: notes
---

<link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.7.2/css/all.css" integrity="sha384-fnmOCqbTlWIlj8LyTjo7mOUStjsKC4pOpQbqyi7RrhN7udi9RwhKkMHpvLbHG9Sr"
  crossorigin="anonymous">


<div class="wrapper-masthead">
  <div class="container">
    <header class="masthead clearfix">
      <a href="/"><font size="22" color="black"><span>&#126;</span></font></a>
      <nav>
        <a href="{{ site.baseurl }}/blog"><font size="5" color="black">~</font>/blog</a>
        <a href="{{ site.baseurl }}/pubs"><font size="5" color="black">~</font>/publications</a> 
        <a href="{{ site.baseurl }}/notes"><font size="5" color="black">~</font>/notes</a>
        <a href="{{ site.baseurl }}/contact"><font size="5" color="black">~</font>/contact</a> 
      </nav>
    </header>
  </div>
</div>


This is a list of summaries of papers related to Robotics/CV. Most of the papers are related to state estimation for autonomous robot navigation or the ones that aid state estimation, the field that I most excited about.

These notes are created for my own understanding and there could be typos, so take them with a pinch of salt in both correctness and conclusions that I draw from them.

## 2021

<style type="text/css">
.tg  {border-collapse:collapse;border-color:#93a1a1;border-spacing:0;}
.tg td{background-color:#fdf6e3;border-bottom-width:1px;border-color:#93a1a1;border-style:solid;border-top-width:1px;
  border-width:0px;color:#002b36;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;
  word-break:normal;}
.tg th{background-color:#657b83;border-bottom-width:1px;border-color:#93a1a1;border-style:solid;border-top-width:1px;
  border-width:0px;color:#fdf6e3;font-family:Arial, sans-serif;font-size:14px;font-weight:normal;overflow:hidden;
  padding:10px 5px;word-break:normal;}
.tg .tg-2bhk{background-color:#efefef;border-color:inherit;text-align:left;vertical-align:top}
.tg .tg-0pky{border-color:inherit;text-align:left;vertical-align:top}
</style>

<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky"><b>#</b></th>
    <th class="tg-0pky"><b>Date</b></th>
    <th class="tg-0pky"><b>Title</b></th>
    <th class="tg-0pky"><b>Description</b></th>
    <th class="tg-0pky"><b>Resources</b></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-2bhk"><font color="green">01</font></td>
    <td class="tg-2bhk">07/01</td>
    <td class="tg-2bhk">Attitude Estimation.</td>
    <td class="tg-2bhk">Discusses estimation of attitude using three different parameterizations (euler, DCM, quaternions). Also discussed is their covariance propagation.</td>
    <td class="tg-2bhk"><a href="https://kvmanohar22.github.io/notes/w01/main.pdf">notes</a>, <a href="https://github.com/kvmanohar22/attitude_estimation">code</a></td>
  </tr>
  <tr>
    <td class="tg-0pky"><font color="green">02</font></td>
    <td class="tg-0pky">15/01</td>
    <td class="tg-0pky">On-Manifold Preintegration Theory for Fast and Accurate Visual-Inertial Navigation.</td>
    <td class="tg-0pky">Discusses an approach for closely-coupled visual inertial navigation.</td>
    <td class="tg-0pky"><a href="https://kvmanohar22.github.io/notes/w02/main.pdf">notes</a></td>
  </tr>
  <tr>
    <td class="tg-2bhk"><font color="green">03</font></td>
    <td class="tg-2bhk">22/01</td>
    <td class="tg-2bhk">Visual-Inertial-Aided Navigation for High-Dynamic Motion in Built Environments Without Initial Conditions.</td>
    <td class="tg-2bhk">Discusses the work that introduced the concept of preintegration of inertial measurements which was an inspiration for the work summarized in <font color="green">02</font>.</td>
    <td class="tg-2bhk"><a href="https://kvmanohar22.github.io/notes/w03/main.pdf">notes</a></td>
  </tr>
</tbody>
</table>
