---
title: "Weighted K-means"
tags:
sidebar: home_sidebar
permalink: weighted-k-means.html
---

\begin{equation}
\newcommand{\b}[1]{\mathbf{#1}}
\newcommand{\bo}[1]{\boldsymbol{#1}} \nonumber
\end{equation}

- $n$：サンプル数
- $c$：クラスタ数
- $\mathbf{x}_k$：特徴ベクトルのサンプル
- $w_k$：サンプルの重み
- $\delta_{ik}$：$k$番目のサンプルをクラスタ$\omega_i$に割り当てたとき1，そうでないとき0
- $\mathbf{p}_i$：各クラスタのプロトタイプ

各$\mathbf{x}_k$に重み$w_k$が付けられている場合のK-meansについて考える．
通常のK-meansと大きな違いはない．

$$
\newcommand{\b}[1]{\mathbf{#1}}
\newcommand{\bo}[1]{\boldsymbol{#1}}
e = \sum_{i = 1}^c \sum_{k = 1}^n \delta_{ik} w_k \|\b{x}_k - \b{p}_i \|^2
$$

---

### 最小化

(1) $\boldsymbol{\delta}_k = \left[ \delta\_{1k}, \ldots, \delta\_{nk} \right]$について最小化

各$\b{p}_i$を固定．

$$
\newcommand{\b}[1]{\mathbf{#1}}
\newcommand{\bo}[1]{\boldsymbol{#1}}
e = \sum_{k = 1}^n \\
e_k := \sum_{i = 1}^c \delta_{ik} w_k \|\b{x}_k - \b{p}_i \|^2
$$

各$e_k$について最小化すれば$e$が最小化される．
よって$j = \argmin_{i} \{ w_k \|\b{x}_k - \b{p}_i \|^2 \}$ならば

$$
\delta_{ik} = \begin{cases}
1 & (i = j) \\
0 & (i \neq j)
\end{cases}
$$

とすることで$e_k$を最小化できる．

---
(2) $\b{p}_i$について最小化

各$\boldsymbol{\delta}_k$を固定．

$$
\newcommand{\b}[1]{\mathbf{#1}}
\newcommand{\bo}[1]{\boldsymbol{#1}}
e = \sum_{i = 1}^c \epsilon_i \\
\epsilon_i := \sum_{k = 1}^n \delta_{ik} w_k \| \b{x}_k - \b{p}_i \|^2
$$

各$\epsilon_i$について最小化すれば$e$が最小化される．
$\frac{\partial \epsilon_i}{\partial \b{p}_i} = 0$とおいて，$\b{p}_i$について解く．

$$
\newcommand{\b}[1]{\mathbf{#1}}
\newcommand{\bo}[1]{\boldsymbol{#1}}
\frac{\partial \epsilon_i}{\partial \b{p}_i} = -2 \sum_{k = 1}^n \delta_{ik} w_k (\b{x}_k - \b{p}_i) = 0 \\
\sum_{k = 1}^n \delta_{ik} w_k \b{x}_k = \b{p}_i \sum_{k = 1}^n \delta_{ik} w_k \\
\therefore \b{p}_i = \frac{\sum_{k = 1}^n \delta_{ik} w_k \b{x}_k}{\sum_{k = 1}^n \delta_{ik} w_k}
$$

---

### アルゴリズム

1. 適当に$c$個のプロトタイプ$\b{p}_1, \ldots, \b{p}_c$を決める．
2. $n$個の各サンプル$\b{x}_k$について，重み付き距離$w_k \|\b{x}_k - \b{p}_i \|^2$が最小のクラスタに割り当てる．
3. 各クラスタ毎に重み付き平均ベクトルを求める．
4. クラスタの再割り当てが発生しなくなるまで2.と3.を繰り返す．
