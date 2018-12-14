---
title:  "線形SVM"
permalink: 2018-12-31-linear-svm.html
sidebar: blog_sidebar
tags: []
---

\begin{align}
\newcommand{\bx}{\mathbf{x}}
\newcommand{\by}{\mathbf{y}}
\newcommand{\bv}{\mathbf{v}}
\newcommand{\bw}{\mathbf{w}}
\newcommand{\balpha}{\boldsymbol{\alpha}}
\newcommand{\bSigma}{\boldsymbol{\Sigma}} \nonumber
\end{align}


SVMを復習しよう．ふとそう思った．今回は最も基本的なケースであるハードマージンかつ線形のSVMについて書く．これは2クラス分類問題において，ある線形判別関数$g(\bw)$によって線形分離可能であることを仮定している．

さて，線形SVMは$g(\bx)$から各クラスに属するサンプルの中で，最も近いサンプルへの距離を最大化するように学習する．
ここからは数式を中心として話を進める．まず2クラスのラベルを$t \in \left\\{+1, -1\right\\}$とする．$g(\bx)$はパラメータ$\bw, b$を用いると$g(\bx) = \bx^T \bw + b$と表現できる．

### 問題設定と定式化

まず，単純な問題設定として2つのクラスのサンプルが線形分離可能だと仮定する．
学習データセットとして$\left\\{(\bx_i, t_i) \right\\}_{i = 1}^n$が与えられたとする．

$g(\bx)$から最も近いサンプルに対する距離は

\begin{align}
\min_{i} \frac{|\bw^T \bx_i + b|}{\\|\bw\\|}
\end{align}

で表される．一方で$g(x)$から2つのクラスに対して最も近いサンプルに対する距離の和は最大化されるようなパラメータを求めたい．つまり
\begin{align}
\argmax_{\bw, b} \left[ \min_{i}  \frac{2|\bw^T \bx_i + b|}{\\|\bw\\|} \right] = \argmax_{\bw, b} \left[ \min_{i} \frac{|\bw^T \bx_i + b|}{\\|\bw\\|} \right]
\end{align}
を求めたい．

ここで，$\forall i \in \left\\{1, \dots, n \right\\}$に対して$ \|\bw^T \bx_i + b\| \geq 1$とすることができる．何故なら，もし最も近いサンプルに対する距離が$\tau$の場合は
\begin{align}
\argmax_{\bw, b} \left[ \min_{i} \tau \frac{|\bw^T \bx_i + b|}{\\|\bw\\|} =  \right] = \argmax_{\bw, b} \left[ \min_{i} \frac{|\bw^T \bx_i + b|}{\\|\bw\\|} \right]
\end{align}
と変形できることから，定数倍したとしても得られる最適解は変わらないからである．これを用いることで，各サンプル$\bx_i$に対して$\bw^T \bx_i + b\ \geq 1$には$t_i = +1$，$\bw^T \bx_i + b\ \leq -1$には$t_i = -1$を割り振るようにすると，
\begin{align}
i \in \left\\{1, \ldots, n \right\\}, \; t_i (\bw^T \bx_i + b) \geq 1
\end{align}
が得られる．

さて，式(4)の変形から最も近いサンプルでは$\|\bw^T \bx + b\| = 1$が成り立つ．これを用いると式(3)は
\begin{align}
\argmax_{\bw, b} \left[ \min_{i} \frac{|\bw^T \bx_i + b|}{\\|\bw\\|} \right] = \argmax_{\bw, b} \frac{1}{\\|\bw\\|} = \argmin_{\bw, b} \\|\bw\\|
\end{align}
とすることができる．最後に今後の計算の準備として$\argmin_{\bw, b} \\|\bw\\| = \argmin_{\bw, b} \frac{1}{2} \\|\bw\\|^2$となることを利用すると，結果的には不等式制約$t_i (\bw^T \bx_i + b) \geq 1$を持つ以下の最適化問題を解けばよいということになる．
\begin{align}
\argmin_{\bw, b} \frac{1}{2} \\|\bw\\|^2
\end{align}

### 最適解の導出

ラグランジュの未定乗数法を用いる．ラグランジュ乗数$\balpha = [\alpha_1, \ldots, \alpha_n]^T \geq \mathbf{0} \; (i = 1, \ldots, N)$を導入するとラグランジュ関数$L(\bw, b, \balpha)$は
\begin{align}
L(\bw, b, \balpha) = \frac{1}{2} \\|\bw\\|^2 - \sum_{i = 1}^n \alpha_i \left\\{t_i (\bw^T \bx_i + b) - 1 \right\\}
\end{align}
となる．$\bw, b$については最小化，$\balpha$については最大化することになる（詳しくは「ラグランジュ双対問題」などで検索）．つまり$\bw, b$については$\frac{\partial L(\bw, b, \balpha)}{\partial \bw} = \mathbf{0}, \frac{\partial L(\bw, b, \balpha)}{\partial b} = 0$を考えれば良い．
\begin{align}
\frac{\partial L(\bw, b, \balpha)}{\partial \bw}
=& \bw - \sum_{i = 1}^n \alpha_i t_i \bx_i = \mathbf{0} \quad \therefore \bw
= \sum_{i = 1}^n \alpha_i t_i \bx_i \newline
\frac{\partial L(\bw, b, \balpha)}{\partial b} 
=& 0 - \sum_{i = 1}^n \alpha_i t_i = 0 \quad \therefore 0 = \sum_{i = 1}^n \alpha_i t_i
\end{align}
これらを用いてラグランジュ関数を変形する．
\begin{align}
L(\balpha) 
&= \frac{1}{2} \\|\bw\\|^2 - \bw^T \underbrace{\sum_{i = 1}^n \alpha_i t_i \bx_i}\_{= \bw} - b \underbrace{\sum_{i = 1}^n \alpha_i t_i}\_{= 0} + \sum_{i = 1}^n \alpha_i = \frac{1}{2} \bw^T \bw - \bw^T \bw + \sum_{i = 1}^n \alpha_i \newline
&= \sum_{i = 1}^n \alpha_i - \frac{1}{2} \bw^T \bw = \sum_{i = 1}^n \alpha_i - \frac{1}{2} \left(\sum_{i = 1}^n \alpha_i t_i \bx_i \right)^T \left(\sum_{j = 1}^n \alpha_j t_j \bx_j \right) \newline
&= \sum_{i = 1}^n \alpha_i - \frac{1}{2} \sum_{i = 1}^n \sum_{j = 1}^n \alpha_i \alpha_j t_i t_j \bx_i^T \bx_j
\end{align}

### 実装例

最急降下法で学習してみる．