---
title: ソフトマージンSVM
tags: [SVM]
sidebar: home_sidebar
permalink: soft-margin-svm.html
---

\begin{align}
\newcommand{\cS}{\mathcal{S}}
\newcommand{\bt}{\mathbf{t}}
\newcommand{\bx}{\mathbf{x}}
\newcommand{\by}{\mathbf{y}}
\newcommand{\bw}{\mathbf{w}}
\newcommand{\bQ}{\mathbf{Q}}
\newcommand{\bphi}{\boldsymbol{\phi}}
\newcommand{\balpha}{\boldsymbol{\alpha}}
\newcommand{\bbeta}{\boldsymbol{\beta}}
\newcommand{\bgamma}{\boldsymbol{\gamma}}
\newcommand{\bxi}{\boldsymbol{\xi}}
\newcommand{\bSigma}{\boldsymbol{\Sigma}} \nonumber
\end{align}

**※事前に[ハードマージンSVM](hard-magin-svm.html)について読んたことを前提として書いているので注意．**

### 問題設定と定式化

ハードマージンSVMは与えられた2つのクラスのデータが線形分離可能だと仮定していた．しかし，一般的に与えられた2クラスのデータが線形分離可能ではない場合が多い．ソフトマージンSVMはそのような問題に対応するSVMで，分類境界によって正しく分類されないサンプルの存在を許容することになる．

より正確に言えば，ハードマージンSVMでは全てのサンプルについて$t_i g(\bx_i) \geq 1$が成り立つとしていたが，ソフトマージンSVMではスラック変数$\xi_i \geq 0$を導入して，$t_i g(\bx_i) + \xi_i \geq 1$が成り立つとしている．ただし$\xi_i$は以下の式によって定める．
\begin{align}
\xi_i = \left\\{ \begin{array}{ll}
    0 & (t_i g(\bx_i) \geq 1) \newline
    |t_i - g(\bx_i)| & (t_i g(\bx_i) < 1)
\end{array} \right.
\end{align}

直感的な理解を得るために$\xi_i$を図示してみる．あるサンプル$\bx_i$のラベルが$t_i = 1$としよう．このとき，$t_i g(\bx_i) \geq 1$となっていれば，マージン境界の外側に位置するためにペナルティを払う必要はないと考えられるので$\xi_i = 0$として良い．しかし下図のように$t_i g(\bx_i) < 1$の場合はマージン境界の内側に位置していることになる．このとき，本来位置してほしいマージン境界までの距離は$\|t_i - g(\bx_i)\|$である．つまり$t_i g(\bx_i) + \|t_i - g(\bx_i)\| = 1$が成り立つ．よって$\xi_i = \|t_i - g(\bx_i)\|$とすることでマージン境界の内側に位置するサンプルであっても$t_i g(\bx_i) + \xi_i = 1$とできるため，全てのサンプルについて$t_i g(\bx_i) + \xi_i \geq 1$である．

![soft-margin-svm](images/soft-margin-svm/soft-margin-svm.png)

$\xi_i$は$\bx_i$がマージン境界の内側にある場合，本来位置してほしいマージン境界までの距離となっているということだったので，その総和$\sum_{i = 1}^n \xi_i$が小さいほうが望ましいということになる．これを踏まえると，不等式制約$t_i g(\bx_i) + \xi_i \geq 1$と$\xi_i \geq 0$を持つ以下の最適化問題を解けば良いということになる．
\begin{align}
\argmin_{\bw, b, \bxi} \frac{1}{2} \\|\bw\\|^2 + C \sum_{i = 1}^n \xi_i
\end{align}
ただし，スラック変数をまとめて$\bxi = [\xi_1, \ldots, \xi_n]^T \geq 0$と表記した．
また，$C$はペナルティパラメータであり，マージンの大きさとペナルティのバランスをとるパラメータである．

### ラグランジュ関数の導出

ラグランジュの未定乗数法を用いる．ラグランジュ乗数$\alpha = [\alpha, \ldots, \alpha_n]^T \geq \mathbf{0}, \bbeta = [\beta_1, \ldots, \beta_n]^T \geq \mathbf{0}$を導入するとラグランジュ関数$L(\bw, b, \bxi, \balpha, \bbeta)$は
\begin{align}
L(\bw, b, \bxi, \balpha, \bbeta) = \frac{1}{2} \\|\bw\\|^2 + C \sum_{i = 1}^n \xi_i - \sum_{i = 1}^n \alpha_i \left\\{t_i (\bw^T \bx_i + b) - 1 + \xi_i \right\\} - \sum_{i = 1}^n \beta_i \xi_i
\end{align}
となる．$\bw, b, \bxi$については最小化，$\balpha, \bbeta$については最大化することになる．$\frac{\partial L}{\partial \bw} = \mathbf{0}, \frac{\partial L}{\partial b} = 0, \frac{\partial L}{\partial \xi_i} = 0$およびKKT条件により，以下が成り立つ．
\begin{align}
\frac{\partial L}{\partial \bw} =& \bw - \sum_{i = 1}^n \alpha_i t_i \bx_i = \mathbf{0} \quad \therefore \bw = \sum_{i = 1}^n \alpha_i t_i \bx_i \newline
\frac{\partial L}{\partial b} =& 0 - \sum_{i = 1}^n \alpha_i t_i = 0 \quad \therefore 0 = \sum_{i = 1}^n \alpha_i t_i \newline
\frac{\partial L}{\partial \xi_i} =& C - \alpha_i - \beta_i = 0 \quad \therefore \alpha_i = C - \beta_i \newline
\alpha_i \geq & 0 \newline
t_i (\bw^T \bx_i + b) - 1 + \xi_i \geq & 0 \newline
\alpha_i \left\\{t_i (\bw^T \bx_i + b) - 1 + \xi_i \right\\} =& 0 \newline
\beta_i \geq & 0 \newline
\xi_i \geq & 0 \newline
\beta_i \xi_i =& 0
\end{align}
$\alpha_i = C - \beta_i$という関係式が得られることから，一方の変数についてのみ最適解を求めれば，もう一方の変数も一意に定まることが分かる．ここでは$\alpha_i$について解くことにする．
これらを用いてラグランジュ関数を変形する．
\begin{align}
L(\balpha) 
&= \frac{1}{2} \\|\bw\\|^2 + C \sum_{i = 1}^n \xi_i - \bw^T \underbrace{\sum_{i = 1}^n \alpha_i t_i \bx_i}\_{= \bw} - b \underbrace{\sum_{i = 1}^n \alpha_i t_i}\_{= 0} + \sum_{i = 1}^n \alpha_i - \sum_{i = 1}^n \underbrace{\alpha_i}\_{=C - \beta_i} \xi_i - \sum_{i = 1}^n \beta_i \xi_i \newline
&= \frac{1}{2} \bw^T \bw + C \sum_{i = 1}^n \xi_i - \bw^T \bw + \sum_{i = 1}^n \alpha_i - \sum_{i = 1}^n (C - \beta_i) \xi_i - \sum_{i = 1}^n \beta_i \xi_i \newline
&= \sum_{i = 1}^n \alpha_i - \frac{1}{2} \bw^T \bw = \sum_{i = 1}^n \alpha_i - \frac{1}{2} \left(\sum_{i = 1}^n \alpha_i t_i \bx_i \right)^T \left(\sum_{j = 1}^n \alpha_j t_j \bx_j \right) \newline
&= \sum_{i = 1}^n \alpha_i - \frac{1}{2} \sum_{i = 1}^n \sum_{j = 1}^n \alpha_i \alpha_j t_i t_j \bx_i^T \bx_j
\end{align}
実はラグランジュ関数はハードマージンSVMと同様の式となり，制約条件のみ異なるという結果が得られる．

### 凸最適化問題への変形

制約条件について改めて確認しよう．式(6), (10)より$0 \leq \alpha_i \leq C$という制約が得られる．次に，ハードマージンSVMにならって，行列$\bQ = (t_i t_j \bx_i^T \bx_j)_{ij} \in \mathbb{R}^{n \times n}$を導入すると
\begin{align}
L(\balpha) = \mathbf{1}^T \balpha - \frac{1}{2} \balpha^T \bQ \balpha
\end{align}
である．

以上より，最適化問題は以下のように整理される．
\begin{alignat}{2}
& \mathrm{maximize} & \quad & \mathbf{1}^T \balpha - \frac{1}{2} \balpha^T \bQ \balpha \newline
& \mathrm{subject\ to} & \quad & \balpha^T \bt = 0, \mathbf{0} \leq \balpha \leq C \mathbf{1}
\end{alignat}

### 参考文献

* [パターン認識と機械学習 下](https://www.maruzen-publishing.co.jp/item/b294551.html)
* [最適化と変分法](https://www.maruzen-publishing.co.jp/item/b294841.html)
