---
title: "ハードマージンSVM"
tags: [SVM]
sidebar: home_sidebar
permalink: hard-margin-svm.html
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
\newcommand{\bSigma}{\boldsymbol{\Sigma}} \nonumber
\end{align}

線形SVMは$g(\bx)$から各クラスに属するサンプルの中で，最も近いサンプルへの距離を最大化するように学習する．ここからは数式を中心として話を進める．まず2クラスのラベルを$t \in \left\\{+1, -1\right\\}$とする．$g(\bx)$はパラメータ$\bw, b$を用いると$g(\bx) = \bw^T \bx + b$と表現できる．

### 問題設定と定式化

ハードマージンSVMでは，単純な問題設定として2つのクラスのサンプルが線形分離可能だと仮定する．学習データセットとして$\left\\{(\bx_i, t_i) \right\\}_{i = 1}^n$が与えられたとする．

$g(\bx)$から最も近いサンプルに対する距離は
\begin{align}
\min_{i} \frac{|\bw^T \bx_i + b|}{\\|\bw\\|}
\end{align}
で表される．一方で$g(\bx)$から2つのクラスに対して最も近いサンプルに対する距離の和は最大化されるようなパラメータを求めたい．つまり
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

### ラグランジュ関数の導出

ラグランジュの未定乗数法を用いる．ラグランジュ乗数$\balpha = [\alpha_1, \ldots, \alpha_n]^T \geq \mathbf{0} \; (i = 1, \ldots, n)$を導入するとラグランジュ関数$L(\bw, b, \balpha)$は
\begin{align}
L(\bw, b, \balpha) = \frac{1}{2} \\|\bw\\|^2 - \sum_{i = 1}^n \alpha_i \left\\{t_i (\bw^T \bx_i + b) - 1 \right\\}
\end{align}
となる．$\bw, b$については最小化，$\balpha$については最大化することになる（詳しくは「ラグランジュ双対問題」などで検索）．ここでKKT条件により，以下が成り立つ．
\begin{align}
\frac{\partial L(\bw, b, \balpha)}{\partial \bw}
=& \bw - \sum_{i = 1}^n \alpha_i t_i \bx_i = \mathbf{0} \quad \therefore \bw
= \sum_{i = 1}^n \alpha_i t_i \bx_i \newline
\frac{\partial L(\bw, b, \balpha)}{\partial b} 
=& 0 - \sum_{i = 1}^n \alpha_i t_i = 0 \quad \therefore 0 = \sum_{i = 1}^n \alpha_i t_i \newline
\alpha_i \geq & 0 \newline
t_i (\bw^T \bx_i + b) - 1 \geq & 0 \newline
\alpha_i \left\\{t_i (\bw^T \bx_i + b) - 1 \right\\} =& 0
\end{align}
これらを用いてラグランジュ関数を変形する．
\begin{align}
L(\balpha) 
&= \frac{1}{2} \\|\bw\\|^2 - \bw^T \underbrace{\sum_{i = 1}^n \alpha_i t_i \bx_i}\_{= \bw} - b \underbrace{\sum_{i = 1}^n \alpha_i t_i}\_{= 0} + \sum_{i = 1}^n \alpha_i = \frac{1}{2} \bw^T \bw - \bw^T \bw + \sum_{i = 1}^n \alpha_i \newline
&= \sum_{i = 1}^n \alpha_i - \frac{1}{2} \bw^T \bw = \sum_{i = 1}^n \alpha_i - \frac{1}{2} \left(\sum_{i = 1}^n \alpha_i t_i \bx_i \right)^T \left(\sum_{j = 1}^n \alpha_j t_j \bx_j \right) \newline
&= \sum_{i = 1}^n \alpha_i - \frac{1}{2} \sum_{i = 1}^n \sum_{j = 1}^n \alpha_i \alpha_j t_i t_j \bx_i^T \bx_j
\end{align}
ちなみに，ラグランジュ係数を用いると$g(\bx)$は
\begin{align}
g(\bx) = \bw^T \bx + b = \left(\sum_{i = 1}^n \alpha_i t_i \bx_i^T \right) \bx + b = \sum_{i = 1}^n \alpha_i t_i \bx_i^T \bx + b
\end{align}
と変形される．

上述のKKT条件において$\alpha_i > 0$に対応するサンプル$\bx_i$はサポートベクトルであり，$t_i g(\bx_i) = 1$を満たす．一方，$\alpha_i = 0$に対応するサンプルはサポートベクトルではない．$\balpha$が求まると，全てのサンプルの添字集合$\left\\{1, \ldots, n \right\\}$の部分集合としてサポートベクトルの添字集合$\cS$が得られる．$\cS$を用いると$g(\bx)$は
\begin{align}
g(\bx) = \sum_{i = 1}^n \alpha_i t_i \bx_i^T \bx + b = \sum_{u \in \cS} \alpha_u t_u \bx_u^T \bx + b
\end{align}
となる．$\\#\cS = n_{\cS}$とし，$\forall i \in \left\\{1, \ldots, n \right\\}, \; t_i^2 = 1$を用いると$b$を以下のようにして求めることができる．
\begin{align}
& \sum_{v \in \cS} t_v^2 g(\bx_v) = \sum_{v \in \cS} g(\bx_v) = \sum_{v \in \cS} t_v \left\\{ t_v g(\bx_v) \right\\} = \sum_{v \in \cS} t_v \newline
& \sum_{v \in \cS} g(\bx_v) = \sum_{v \in \cS} \left( \sum_{u \in \cS} \alpha_u t_u \bx_u^T \bx_v + b \right) = \sum_{v \in \cS} \left( \sum_{u \in \cS} \alpha_u t_u \bx_u^T \bx_v \right) + n_{\cS} b \newline
& \therefore b = \frac{1}{n_{\cS}} \sum_{v \in \cS} \left(t_v - \sum_{u \in \cS} \alpha_u t_u \bx_u^T \bx_v \right)
\end{align}

### 凸最適化問題への変形

さて，以上でラグランジュ関数の導出は出来た．ここからは実際に凸最適化問題として解くことを考える．  

まず，導出したラグランジュ関数が$\balpha$についての二次計画問題となっていることが分かるように変形してみる．行列$\bQ = (t_i t_j \bx_i^T \bx_j)_{ij} \in \mathbb{R}^{n \times n}$を導入すると
\begin{align}
L(\balpha) = \mathbf{1}^T \balpha - \frac{1}{2} \balpha^T \bQ \balpha
\end{align}
である．さらに$\bt = [t_1, \ldots, t_n]^T$と表記すると，式(9)の等式制約は$\balpha^T \bt = 0$と書ける．

以上より，最適化問題は以下のように整理される．
\begin{alignat}{2}
& \mathrm{maximize} & \quad & \mathbf{1}^T \balpha - \frac{1}{2} \balpha^T \bQ \balpha \newline
& \mathrm{subject\ to} & \quad & \balpha^T \bt = 0, \balpha \geq \mathbf{0}
\end{alignat}

### 非線形変換とカーネルトリック

ここまでは入力変数$\bx$はそのまま用いていたが，入力空間では線形分離可能でない場合がある．そのような場合でも，特徴空間への写像$\bphi(\bx)$を用いることで，線形分離可能になる場合がある．$\bphi(\bx)$による写像を用いる場合は，$\bx$の箇所を単純に$\bphi(\bx)$に置き換えるだけで良い．また，$L(\balpha)$や$\bQ$中に現れる内積$\bphi(\bx_i)^T\bphi(\bx_j)$はカーネル$k(\bx_i, \bx_j)$で置き換えることができる．

### 参考文献

* [パターン認識と機械学習 下](https://www.maruzen-publishing.co.jp/item/b294551.html)
* [最適化と変分法](https://www.maruzen-publishing.co.jp/item/b294841.html)
