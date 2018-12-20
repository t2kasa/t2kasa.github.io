---
title: On the Algorithmic Implementation of Multiclass Kernel-based Vector Machines
tags: [SVM]
sidebar: home_sidebar
permalink: crammer-and-singer-mcsvm.html
---

\begin{align}
\newcommand{\cS}{\mathcal{S}}
\newcommand{\cX}{\mathcal{X}}
\newcommand{\cY}{\mathcal{Y}}
\newcommand{\be}{\mathbf{e}}
\newcommand{\bt}{\mathbf{t}}
\newcommand{\bx}{\mathbf{x}}
\newcommand{\by}{\mathbf{y}}
\newcommand{\bw}{\mathbf{w}}
\newcommand{\bW}{\mathbf{W}}
\newcommand{\bQ}{\mathbf{Q}}
\newcommand{\btau}{\boldsymbol{\tau}}
\newcommand{\bphi}{\boldsymbol{\phi}}
\newcommand{\balpha}{\boldsymbol{\alpha}}
\newcommand{\bbeta}{\boldsymbol{\beta}}
\newcommand{\bgamma}{\boldsymbol{\gamma}}
\newcommand{\bxi}{\boldsymbol{\xi}}
\newcommand{\bSigma}{\boldsymbol{\Sigma}} \nonumber
\end{align}

### 概要

Crammer and Singerによる多クラスSVMのアプローチを紹介する．論文中の双対問題導出までの式変形をなぞっている．  
効率的な学習方法についても述べられているが，そこまでは読めていない＆本ページに書いていないので注意．**もしそちらに興味がある方がいれば，論文を読んだまとめ資料を是非公開して頂きたいです．**

### 前準備

学習データとして$S = \left\\{(\bx_1, y_1), \ldots, (\bx_m, y_m) \right\\}$が与えられ，$\bx_i \in \cX \subseteq \mathbb{R}^n, y_i \in \cY = \left\\{1, \ldots, k \right\\}$とする．多クラス分類するための分類器として$H: \cX \rightarrow \cY$を学習したい．本アプローチでは$H$を以下のように定義する．
\begin{align}
H_{\bW}(\bx) = \argmax_{\gamma \in \cY} \bw_{\gamma}^T \bx
\end{align}
ここで$\bW \in \mathbb{R}^{k \times n}$であり，$\bw_{\gamma}^T \in \mathbb{R}^n$は$\bW$の$\gamma$行目にあたる．直感的には，各クラス$\gamma \in \cY$に対応するベクトル$\bw_{\gamma}$と入力$\bx$の内積を$\bx$が$\gamma$に属するスコアとみなしていると考えることができる（あくまでスコアであって確率ではないので注意）．

### 主問題

制約条件について見ていこう．まず線形分離可能，即ちハードマージンと仮定して考えてみる．
上述したスコアという考え方を用いると，ペア$\left\\{(\bx_i, y_i) \right\\}$に対応するスコア$\bw_{y_i}^T \bx_i$が他のクラスに対するスコアよりも大きくなることが望ましいといえる．つまり
\begin{align}
\forall i, \gamma \quad \bw_{y_i}^T \bx_i + \delta_{y_i, \gamma} - \bw_{\gamma}^T \bx_i \geq 1
\end{align}
が成り立つとする．ただし$\delta_{y_i, \gamma}$は$y_i = \gamma$のとき$1$であり，$y_i \neq \gamma$のとき$0$である．これは$y_i = \gamma$のときにも不等式が成り立つようにするためである．

次に，スラック変数$\xi_i \geq 0$を導入してソフトマージンに対応した場合を考える．これは通常の2クラスSVMと同様に
\begin{align}
\forall i, \gamma \quad \bw_{y_i}^T \bx_i + \delta_{y_i, \gamma} - \bw_{\gamma}^T \bx_i \geq 1 - \xi_i
\end{align}
とすればよい．ちなみに$y_i = \gamma$の場合は単に$\xi_i \geq 0$が得られる．  
また[#スラック変数についての補足](#スラック変数についての補足)も参照してほしい．

以上より，主問題は
\begin{alignat}{2}
& \mathrm{minimize} & \quad & \frac{1}{2} \beta \\|\bW\\|\_2^2 + \sum_{i = 1}^m \xi_i \newline
& \mathrm{subject\ to} & \quad & \forall i, \gamma \quad \bw_{y_i}^T \bx_i + \delta_{y_i, \gamma} - \bw_{\gamma}^T \bx_i \geq 1 - \xi_i
\end{alignat}
である．

### 双対問題の導出

ラグランジュ変数$\eta_{i, \gamma} \geq 0$を導入するとラグランジュ関数$L(\bW, \left\\{\xi_i \right\\}, \left\\{\eta_{i, \gamma} \right\\})$が得られる．
\begin{align}
L(\bW, \left\\{\xi_i \right\\}, \left\\{\eta_{i, \gamma} \right\\}) = \frac{1}{2} \beta \sum_{\gamma} \\|\bw_{\gamma}\\|\_2^2 + \sum_i \xi_i + \sum_{i, \gamma} \eta_{i, \gamma} \left[\ \bw_{\gamma}^T \bx_i - \delta_{y_i, \gamma} - \bw_{y_i}^T \bx_i + 1 - \xi_i \right]
\end{align}
$\frac{\partial L}{\partial \xi_i} = 0$より
\begin{align}
\frac{\partial L}{\partial \xi_i} = 1 - \sum_{\gamma} \eta_{i, \gamma} = 0 \; \Rightarrow \; \sum_{\gamma} \eta_{i, \gamma} = 1 \newline
\end{align}
を得る．これを用いると
\begin{align}
\sum_{i, \gamma} \eta_{i, \gamma} \bw_{y_i}^T \bx_i = \sum_i \bw_{y_i}^T \bx_i \underbrace{ \left( \sum_{\gamma} \eta_{i, \gamma} \right) }\_{=1} = \sum_i \bw_{y_i} \bx_i
\end{align}
となる．$\frac{\partial}{\partial \bw_r} \bw_{y_i} \bx_i = \delta_{y_i, \gamma} \bx_i$となることに留意すると，$\frac{\partial L}{\partial \bw_r} = \mathbf{0}$より
\begin{align}
\frac{\partial L}{\partial \bw_r} = \beta \bw_r - \left[\sum_{i} \delta_{y_i, \gamma} \bx_i - \sum_{i} \eta_{i, \gamma} \bx_i \right] = \mathbf{0} \; \Rightarrow \; \bw_r = \beta^{-1} \left[\sum_{i} \left( \delta_{y_i, \gamma} - \eta_{i, \gamma} \right) \bx_i \right]
\end{align}
である．

さて，以上の結果を用いて双対問題を導出しよう．ラグランジュ関数を整理して変数$\left\\{\eta_{i, \gamma} \right\\}$のみにした関数を$Q(\eta)$と表記することにする．$Q(\eta)$は
\begin{align}
Q(\eta) 
=& \frac{1}{2} \beta \sum_{\gamma} \\|\bw_{\gamma}\\|\_2^2 + \sum_i \xi_i + \sum_{i, \gamma} \eta_{i, \gamma} \bw_{\gamma}^T \bx_i - \sum_{i, \gamma} \eta_{i, \gamma} \delta_{y_i, \gamma} - \sum_{i, \gamma} \eta_{i, \gamma} \bw_{y_i}^T \bx_i \nonumber \newline
& + \sum_{i, \gamma} \eta_{i, \gamma} - \sum_{i, \gamma} \eta_{i, \gamma} \xi_i \newline
=& \frac{1}{2} \beta \sum_{\gamma} \\|\bw_{\gamma}\\|\_2^2 + \sum_i \xi_i + \sum_{i, \gamma} \eta_{i, \gamma} \bw_{\gamma}^T \bx_i - \sum_{i, \gamma} \eta_{i, \gamma} \delta_{y_i, \gamma} - \sum_{i, \gamma} \eta_{i, \gamma} \bw_{y_i}^T \bx_i \nonumber \newline
& + \sum_i \underbrace{\left( \sum_{\gamma} \eta_{i, \gamma} \right) }\_{=1} - \sum_i \xi_i \underbrace{\left( \sum_{\gamma} \eta_{i, \gamma} \right) }\_{=1} \newline
=& \underbrace{\sum_{i, \gamma} \eta_{i, \gamma} \bw_{\gamma}^T \bx_i}\_{:=S_1} - \underbrace{\sum_{i, \gamma} \eta_{i, \gamma} \bw_{y_i}^T \bx_i}\_{:=S_2} + \underbrace{\frac{1}{2} \beta \sum_{\gamma} \\|\bw_{\gamma}\\|\_2^2}\_{:=S_3} + \underbrace{\sum_i 1}\_{\mathrm{const.}} - \sum_{i, \gamma} \eta_{i, \gamma} \delta_{y_i, \gamma} \newline
\end{align}
となる．まず，$\mathrm{const.}$の箇所は最適化問題としては影響を与えないのでこれ以降考えないものとする．また，$S_1, S_2, S_3$は以降の変形を一旦個別に確認するためである．さて，$S_1, S_2, S_3$をそれぞれ変形していこう．
\begin{align}
S_1 =& \sum_{i, \gamma} \eta_{i, \gamma} \bx_i^T \left[ \beta^{-1} \sum_{j} \left( \delta_{y_j, \gamma} - \eta_{j, \gamma} \right) \bx_j \right] = \beta^{-1} \sum_{i, j} \bx_i^T \bx_j \sum_{\gamma} \eta_{i, \gamma} \left( \delta_{y_j, \gamma} - \eta_{j, \gamma} \right) \newline
S_2 =& \sum_{i, \gamma} \eta_{i, \gamma} \bx_i^T \left[ \beta^{-1} \sum_{j} \left( \delta_{y_j, y_i} - \eta_{j, y_i} \right) \bx_j \right] = \beta^{-1} \sum_{i, j} \bx_i^T \bx_j \left( \delta_{y_j, y_i} - \eta_{j, y_i} \right) \underbrace{\sum_{\gamma} \eta_{i, \gamma}}\_{=1} \newline
=& \beta^{-1} \sum_{i, j} \bx_i^T \bx_j \left( \delta_{y_j, y_i} - \eta_{j, y_i} \right) = \beta^{-1} \sum_{i, j} \bx_i^T \bx_j \sum_{\gamma} \delta_{y_i, \gamma} \left( \delta_{y_j, \gamma} - \eta_{j, \gamma} \right) \newline
S_3 =& \frac{1}{2} \beta \sum_{\gamma} \bw_{\gamma}^T \bw_{\gamma} = \frac{1}{2} \beta \sum_{\gamma} \left[ \beta^{-1} \sum_{i} \left( \delta_{y_i, \gamma} - \eta_{i, \gamma} \right) \bx_i \right]^T \left[ \beta^{-1} \sum_{j} \left( \delta_{y_j, \gamma} - \eta_{j, \gamma} \right) \bx_j \right] \newline
=& \frac{1}{2} \beta^{-1} \sum_{i, j} \bx_i^T \bx_j \sum_{\gamma} \left(\delta_{y_i, \gamma} - \eta_{i, \gamma} \right) \left( \delta_{y_j, \gamma} - \eta_{j, \gamma} \right) \newline
S_1 - S_2 =& - \beta^{-1} \sum_{i, j} \bx_i^T \bx_j \sum_{\gamma} \left(\delta_{y_i, \gamma} - \eta_{i, \gamma} \right) \left( \delta_{y_j, \gamma} - \eta_{j, \gamma} \right)
\end{align}
これらの結果を用いると$Q(\eta)$は
\begin{align}
Q(\eta) 
=& S_1 - S_2 + S_3 - \sum_{i, \gamma} \eta_{i, \gamma} \delta_{y_i, \gamma} \newline
=& - \beta^{-1} \sum_{i, j} \bx_i^T \bx_j \sum_{\gamma} \left(\delta_{y_i, \gamma} - \eta_{i, \gamma} \right) \left( \delta_{y_j, \gamma} - \eta_{j, \gamma} \right) \newline
& + \frac{1}{2} \beta^{-1} \sum_{i, j} \bx_i^T \bx_j \sum_{\gamma} \left(\delta_{y_i, \gamma} - \eta_{i, \gamma} \right) \left( \delta_{y_j, \gamma} - \eta_{j, \gamma} \right) - \sum_{i, \gamma} \eta_{i, \gamma} \delta_{y_i, \gamma} \newline
=& - \frac{1}{2} \beta^{-1} \sum_{i, j} \bx_i^T \bx_j \sum_{\gamma} \left(\delta_{y_i, \gamma} - \eta_{i, \gamma} \right) \left( \delta_{y_j, \gamma} - \eta_{j, \gamma} \right) - \sum_{i, \gamma} \eta_{i, \gamma} \delta_{y_i, \gamma}
\end{align}
となり，かなりすっきりした形になる．

もう少しだけ整理しよう．$i$番目の成分が$1$で他の成分が$0$のベクトル（いわゆるone-hot）を$\be_i$と表記することにする．また，$\boldsymbol{\eta}\_i = \left[ \eta_{1, 1}, \ldots, \eta_{1, k} \right]^T \geq \mathbf{0}$とする．更に$\boldsymbol{\tau}\_i = \be_{y_i} - \boldsymbol{\eta}\_i$とすると
\begin{align}
Q(\eta) 
=& - \frac{1}{2} \beta^{-1} \sum_{i, j} \bx_i^T \bx_j \left[ \left(\be_{y_i} - \boldsymbol{\eta}\_i \right)^T \left(\be_{y_j} - \boldsymbol{\eta}\_j \right) \right] - \sum_i \boldsymbol{\eta}\_{i}^T \be_{y_i} \newline
=& - \frac{1}{2} \beta^{-1} \sum_{i, j} \bx_i^T \bx_j \left[ \btau_i^T \btau_j \right] - \sum_i \left( \be_{y_i} - \btau_i \right)^T \be_{y_i} \newline
=& - \frac{1}{2} \beta^{-1} \sum_{i, j} \bx_i^T \bx_j \left[ \btau_i^T \btau_j \right] - \sum_i \left(\underbrace{1}\_{\mathrm{const.}} - \btau_i^T \be_{y_i} \right)
\end{align}
となる．ここで$\boldsymbol{\eta}\_i \geq 0$および$\btau_i = \be_{y_i} - \boldsymbol{\eta}\_i$から不等式制約$\btau_i \leq \be_{y_i}$が得られる．これに加えて，$\sum_{\gamma} \eta_{i, \gamma} = \boldsymbol{\eta}\_i^T \mathbf{1} = 1$であることから$\btau_i^T \mathbf{1} = \be_{y_i}^T \mathbf{1} - \boldsymbol{\eta}\_i^T \mathbf{1} = 1 - 1 = 0$という等式制約が得られる．最後に定数項$1$を無視すると，双対問題は
\begin{alignat}{2}
& \mathrm{maximize} & \quad & - \frac{1}{2} \sum_{i, j} \bx_i^T \bx_j \left[ \btau_i^T \btau_j \right] + \beta \sum_i \btau_i^T \be_{y_i} \newline
& \mathrm{subject\ to} & \quad & \forall i \quad \btau_i^T \mathbf{1} = 0, \btau_i \leq \be_{y_i}
\end{alignat}
となる．また$\btau_i$を用いることで$H_{\bW}(\bx)$は
\begin{align}
H_{\bW}(\bx) = \argmax_{\gamma \in \cY} \bw_{\gamma}^T \bx = \argmax_{\gamma \in \cY} \left[ \sum_i \tau_{i, \gamma} \bx_i^T \bx \right]
\end{align}
と表現できる．

-->

### 最適化問題の分解

得られた双対問題は凸二次計画問題となっているが，$mk$個の変数があるため，そのまま解くには$mk \times mk$という非常に大きな行列が必要になってしまう．そこで，本論文では以下の方法によって小さな問題に分解することで効率的に解くことを考える．

変数を$m$個のdisjoint sets $\left\\{ \btau_i \| \btau_i \leq \be_{y_i}, \btau_i^T \mathbf{1} = 0 \right\\}$に分割する．そして，あるインデックス$p$を選択し，$\btau_p$のみを変数として扱う最適化問題を解く，という手順を繰り返す．$Q(\tau)$を$\btau_p$の関数として変形してみると
\begin{align}
Q_p(\btau_p) 
=& - \frac{1}{2} \sum_{i, j} \bx_i^T \bx_j \left[ \btau_i^T \btau_j \right] + \beta \sum_i \btau_i^T \be_{y_i} \newline
=& - \frac{1}{2} \bx_p^T \bx_p \left[ \btau_p^T \btau_p \right] - \sum_{i \neq p} \bx_p^T \bx_i \left[ \btau_p^T \btau_i \right] - \frac{1}{2} \sum_{i \neq p, j \neq p} \bx_i^T \bx_j \left[ \btau_i^T \btau_j \right] + \beta \btau_p^T \be_{y_p} + \beta \sum_{i \neq p} \btau_i^T \be_{y_i} \newline
=& - \frac{1}{2} \bx_p^T \bx_p \left[ \btau_p^T \btau_p \right] - \btau_p^T \left[ - \beta \be_{y_p} + \sum_{i \neq p} \bx_p^T \bx_i \btau_i \right] + \left[ - \frac{1}{2} \sum_{i \neq p, j \neq p} \bx_i^T \bx_j \left[ \btau_i^T \btau_j \right] + \beta \sum_{i \neq p} \btau_i^T \be_{y_i} \right]\end{align}
となる．$A_p = \bx_p^T \bx_p, B_p = - \beta \be_{y_p} + \sum_{i \neq p} \bx_p^T \bx_i \btau_i, C_p = - \frac{1}{2} \sum_{i \neq p, j \neq p} \bx_i^T \bx_j \left[ \btau_i^T \btau_j \right] + \beta \sum_{i \neq p} \btau_i^T \be_{y_i}$とおくと$Q_p(\tau_p) = - \frac{1}{2} A_p \btau_p^T \btau_p - B_p^T \btau_p + C_p$である．定数項を無視すると，最適化問題は
\begin{alignat}{2}
& \mathrm{minimize} & \quad & Q_p(\btau_p) =\frac{1}{2} A_p \btau_p^T \btau_p + B_p^T \btau_p \newline
& \mathrm{subject\ to} & \quad & \btau_p \leq \be_{y_p}, \btau_p^T \mathbf{1} = 0
\end{alignat}
となる（符号を反転して最小化にしていることに注意）．この最適化問題を解く方法として不動点アルゴリズムによる方法が詳述されている．

学習アルゴリズムは以下の図2のように表現できる．

![fig2](images/figs/crammer-and-singer-mcsvm/fig2.png){:style="border: 1px solid black"}

### References

* [Koby Crammer, Yoram Singer. On the Algorithmic Implementation of Multiclass Kernel-based Vector Machines. JMLR, 2001](http://jmlr.csail.mit.edu/papers/v2/crammer01a.html){:target="_blank"}

---

### 補足等
#### スラック変数についての補足

本手法ではスラック変数の数はサンプル数の$m$になる．不等式制約は
\begin{align}
\forall i, \gamma \quad \bw_{y_i}^T \bx_i + \delta_{y_i, \gamma} - \bw_{\gamma}^T \bx_i \geq 1 - \xi_i
\end{align}
となっているから，先に$i$を定めると$\forall \gamma$に対してこの不等式が成り立つような$\xi_i$を決める必要があることが分かる．これは
\begin{align}
\xi_i = \max_{\gamma} \left[ 1 - \delta_{y_i, \gamma} + \bw_{\gamma}^T \bx_i - \bw_{y_i}^T \bx_i \right]
\end{align}
と定めれば成り立つことが分かる．これは$y_i \neq \argmax_{\gamma} \left[ \bw_{\gamma}^T \bx_i \right]$の場合には，最もペナルティが大きくなるようなクラスについてのスコアのみ着目していると解釈することができる．
