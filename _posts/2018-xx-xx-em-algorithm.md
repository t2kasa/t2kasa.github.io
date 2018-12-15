---
title:  "EMアルゴリズムの復習"
permalink: 2018-12-12-em-algorithm.html
sidebar: blog_sidebar
tags: []
---

\begin{align}
\newcommand{\btheta}{\boldsymbol{\theta}}
\newcommand{\bx}{\mathbf{x}}
\newcommand{\bX}{\mathbf{X}}
\newcommand{\bz}{\mathbf{z}}
\newcommand{\bZ}{\mathbf{Z}} \nonumber
\end{align}

PRML9章より，一般のEMアルゴリズムを復習する．まず，使用する記号を以下のように決めておく．

* 観測変数の集合：$\bX$
* 潜在変数の集合：$\bZ$
* パラメータの集合：$\btheta$

さて，一般に潜在変数は観測できないため，観測変数の集合のみからパラメータ$\btheta$を推定することになる．即ち，尤度関数
\begin{align}
p(\bX|\btheta) = \sum_{\bZ} p(\bX, \bZ \| \btheta)
\end{align}
を最大化するパラメータを求めたい．

EMアルゴリズムは$p(\bX \| \btheta)$を求めるのは困難だが，$p(\bX, \bZ \| \btheta)$を求めるのが容易な場合に有効である．ここで，


さて，観測変数・潜在変数の同時確率分布$p(\bX, \bZ \| \btheta)$を考える．

EMアルゴリズムで対象とする