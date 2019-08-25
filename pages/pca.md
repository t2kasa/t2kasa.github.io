---
title: PCA
tags: []
sidebar: home_sidebar
permalink: pca.html
---

$d$次元から$\tilde{d}$次元にする．

$\tilde{d}$次元部分空間 $\left\\{\mathbf{u}_1, \ldots, \mathbf{u}\_{\tilde{d}} \right\\}, \mathbf{u}_i \in \mathbb{R}^d, \mathbf{u}_i^T \mathbf{u}_j = \delta\_{ij}$

元の特徴空間から部分空間への変換行列$\mathbf{A}$は
$\mathbf{A} = \left[\mathbf{u}_1, \ldots, \mathbf{u}\_{\tilde{d}} \right]$

\begin{align}
\mathbf{y} = \mathbf{A}^T \mathbf{x} \in \mathbb{R}^{\tilde{d}}
\end{align}

部分空間での平均$\tilde{\mathbf{m}}$は元の特徴空間の平均を$\mathbf{m}$とすると

\begin{align}
\tilde{\mathbf{m}} = \frac{1}{n} \sum_{\mathbf{y} \in \mathcal{Y}} \mathbf{y} = \frac{1}{n} \sum_{\mathbf{x} \in \mathcal{X}} \mathbf{A}^T \mathbf{x} = \mathbf{A}^T \frac{1}{n} \sum_{\mathbf{x} \in \mathcal{X}} \mathbf{x} = \mathbf{A}^T \mathbf{m}
\end{align}

部分空間での分散$\tilde{\sigma}^2 (\mathbf{A})$は，元の特徴空間の共分散行列を$\mathbf{\Sigma}$とすると

\begin{align}
\tilde{\sigma}^2 (\mathbf{A}) 
&= \frac{1}{n} \sum_{\mathbf{y} \in \mathcal{Y}} (\mathbf{y} - \tilde{\mathbf{m}})^T (\mathbf{y} - \tilde{\mathbf{m}}) = \frac{1}{n} \sum_{\mathbf{y} \in \mathcal{Y}} \left\\{\mathbf{A}^T (\mathbf{x} - \mathbf{m}) \right\\}^T \left\\{\mathbf{A}^T (\mathbf{x} - \mathbf{m}) \right\\} \newline
&= \frac{1}{n} \sum_{\mathbf{y} \in \mathcal{Y}} \mathrm{tr} \left( \left\\{\mathbf{A}^T (\mathbf{x} - \mathbf{m}) \right\\} \left\\{\mathbf{A}^T (\mathbf{x} - \mathbf{m}) \right\\}^T \right) 
= \frac{1}{n} \sum_{\mathbf{y} \in \mathcal{Y}} \mathrm{tr} \left( \mathbf{A}^T (\mathbf{x} - \mathbf{m}) (\mathbf{x} - \mathbf{m})^T \mathbf{A} \right) \newline
&= \mathrm{tr} (\mathbf{A}^T \mathbf{\Sigma} \mathbf{A})
\end{align}

よって解きたい最適化問題は以下のように表せる．

\begin{alignat}{2}
& \mathrm{maximize} & \quad & \mathrm{tr} (\mathbf{A}^T \mathbf{\Sigma} \mathbf{A}) \newline
& \mathrm{subject\ to} & \quad & \mathbf{A}^T \mathbf{A} - \mathbf{I} = \mathbf{O}
\end{alignat}

制約について．$\mathbf{A}^T \mathbf{A}$の対角成分が全て1になるという制約になっているので，ラグランジュ乗数$\lambda_1, \ldots, \lambda_\tilde{d}$を導入すると，最適化問題は

\begin{align}
J(\mathbf{A}) = \mathrm{tr} (\mathbf{A}^T \mathbf{\Sigma} \mathbf{A}) - \sum_{i = 1}^\tilde{d} \left( (\mathbf{A}^T \mathbf{A})_{ii} - 1 \right) \lambda_i = \mathrm{tr} (\mathbf{A}^T \mathbf{\Sigma} \mathbf{A}) - \mathrm{tr} ((\mathbf{A}^T\mathbf{A} - \mathbf{I}) \mathbf{\Lambda})
\end{align}

となる．$\mathrm{tr} (\cdot)$では行列の積の順番を入れ替えても値は変わらないので，$\mathrm{tr} ((\mathbf{A}^T\mathbf{A} - \mathbf{I}) \mathbf{\Lambda}) = \mathrm{tr} (\mathbf{A}^T \mathbf{\Lambda} \mathbf{A} - \mathbf{\Lambda})$と変形する．

\begin{align}
\frac{\partial}{\partial \mathbf{A}} J(\mathbf{A}) = 2 \mathbf{\Sigma} \mathbf{A} - 2 \mathbf{A} \mathbf{\Lambda} = 0 \newline
\mathbf{A}^T \mathbf{\Sigma} \mathbf{A} = \mathbf{\Lambda}
\end{align}

よって$\mathbf{A}$は$\mathbf{\Sigma}$を対角化する行列．

\begin{align}
\mathrm{tr} (\mathbf{A}^T \mathbf{\Sigma} \mathbf{A}) = \mathrm{tr} (\mathbf{\Lambda}) = \sum_{i = 1}^\tilde{d} \lambda_i
\end{align}
