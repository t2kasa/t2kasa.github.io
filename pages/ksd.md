---
title: Kernelized Stein Discrepancy
tags: [Work In Progress]
sidebar: home_sidebar
permalink: ksd.html
---

\begin{equation}
\newcommand{\calA}{\mathcal{A}}
\newcommand{\calF}{\mathcal{F}}
\newcommand{\calH}{\mathcal{H}}
\newcommand{\bbE}{\mathbb{E}}
\end{equation}

[A Short Introduction to Kernelized Stein Discrepancy](http://www.cs.dartmouth.edu/~qliu/PDF/ksd_short.pdf)  
Kernelized Stein Discrepancyについて現状わかっている範囲でメモ。

### Stein's Identity

微分可能な確率密度関数$p(x)$と関数$f(x) \in \mathbb{R}$があり，$\lim_{\parallel x \parallel \rightarrow \infty} p(x)f(x) = 0$のとき

\begin{align}
\mathbb{E}_{x \sim p} \left[f(x) \nabla_x \log p(x) + \nabla_x f(x) \right] = 0, \hspace{10pt} \forall f
\end{align}

を満たす．

以降，表記を容易にするため，$\calA_p = f(x) \nabla_x \log p(x) + \nabla_x f(x)$とする．
$\calA_p$は重要な性質として線形性があり，$\calA_p(f + g) = \calA_p f + \calA_p g$とできる．

### Stein Discrepancy

$\calA_p f(x)$を$p(x)$での期待値ではなく，異なる確率分布$q(x)$で期待値をとることを考える．

\begin{equation}
\bbE_{x \sim q} \left[\calA_p f(x) \right]
\end{equation}

この期待値は$q(x)$での期待値なので，0にはならないような$f$が存在する．
上式は
\begin{align}
\bbE_{x \sim q} \left[\calA_p f(x) \right] 
&= \bbE_{x \sim q} \left[\calA_p f(x) \right] - \bbE_{x \sim q} \left[\calA_q f(x) \right] \hspace{10pt} \left(\because \bbE_{x \sim q} \left[\calA_q f(x) \right] = 0 \right) \newline
&= \bbE_{x \sim q} \left[f(x) \left(\nabla_x \log p(x) - \nabla_x \log q(x) \right) \right]
\end{align}
と変形すると，$\nabla_x \log p(x) - \nabla_x \log q(x) = 0$，即ち$p = q$の場合に限り上式は0になる．この方法によって，Stein's identityは2つの分布を比較している．

すると，qに従うサンプル集合$\left\\{x_i\right\\}$が与えられると$p = q$のときはサンプルでの平均$\sum_i \calA_p f(x_i) / n$が0に近くなる．これによってデータ$\left\\{x_i\right\\}$とモデル$p$がどれぐらい一致しているかを$p$の正規化定数を計算せずに測ることができる．

$f$については最もStein's identityから遠ざかるようにすることを考える．
即ち$f$については最大化する．このmetricを**Stein discrepancy**と呼ぶ．
\begin{align}
\sqrt{S(q, p)} = \max_{f \in \calF} \bbE_{x \sim q} \left[\calA_p f(x) \right]
\end{align}

### Solving the optimization

既知の基底関数$f_i(x)$の集合と未知の係数$w_i$の線形結合で$f(x)$を表すとする．
即ち，$f(x) = \sum_i w_i f_i(x)$．すると，$\calA_p$と$\bbE$は線形性を持つので
\begin{align}
\bbE_{x \sim q} \left[\calA_p f \right] 
&= \bbE_{x \sim q} \left[\calA_p \sum_i w_i f_i(x) \right] \newline
&= \bbE_{x \sim q} \left[\sum_i w_i \calA_p f_i(x) \right] \newline
&= \sum_i w_i \bbE_{x \sim q} \left[\calA_p f_i(x) \right] \newline
&= \sum_i w_i \beta_i \quad \mathrm{where} \quad \beta_i = \bbE_{x \sim q} \left[\calA_p f_i(x) \right]
\end{align}
と変形できる．
すると，以下の目的関数は容易に最適解が求まる．
\begin{align}
\sum_i w_i \beta_i, \quad s.t. \quad \\|w\\| \leq 1
\end{align}
最適解は$w_i = \beta_i / \\|\beta_i\\|$．

上記の最適化問題は，$f(x)$が既知の基底関数の重み付き線形結合で表現される場合である．
この場合，Stein discrepancyを容易に求めることができるが，$\calF$を著しく制限していることになる．そこでkernel methodとStein's discrepancyを組み合わせた**kernelized Stein discrepancy (KSD)**が提案された．

$\calF$をreproducing kernel Hilbert space (RKHS) $\calH$の単位球とする．
$\calH$に関連付けられた正定値カーネルを$k(x, x^{\prime})$とするとKSDは次式で定義される．

\begin{align}
\sqrt{\mathbb{S}(q, p)} = \max_{f \in \calF} \left\\{ \bbE_{x \sim q} \left[\calA_p f(x) \right], \;\; s.t. \;\; \\|f\\|_{\calH} \leq 1 \right\\}
\end{align}

再生性より$f(x) = \langle f(\cdot), k(x, \cdot) \rangle_{\calH}$であるが，さらに$\nabla_x f(x) = \langle f(\cdot), \nabla_x k(x, \cdot) \rangle_{\calH}$が成り立つ．
よって
\begin{align}
\bbE_{x \sim q} \left[\calA_p f(x) \right] = \langle f(\cdot), \bbE_{x \sim q} \left[\calA_p k(\cdot, x) \right] \rangle_{\calH}
\end{align}
となる．$\beta_{q, p}(\cdot) = \bbE_{x^{\prime} \sim q} \left[\calA_p k(\cdot, x) \right]$を定義すると，KSDは
\begin{align}
\max_{f} \langle f, \beta_{q, p} \rangle_{\calH}, \;\; s.t. \;\; \\|f\\|_{\calH} \leq 1
\end{align}
となる．

さて，KSD導入前に$f(x)$が線形結合で表現される場合の最適化問題は容易に解けることを確認した．
線形結合は係数と基底関数についての内積とみなすことができるので，KSDも同様に考えることができる．
したがって最適な$f$は$f = \beta_{q, p} / \|\beta_{q, p}\|\_{\mathcal{H}}$であり，$\mathcal{S}(q, p) = \\|\beta_{q, p}\\|_{\mathcal{H}}^2$．
さらに，以下の式を得ることができる．

\begin{align}
\mathcal{S}(q, p) = \bbE_{x, x^{\prime} \sim q} \left[\kappa_p (x, x^{\prime}) \right], \quad \mathrm{where} \quad \kappa_p (x, x^{\prime}) = \calA_p^x \calA_p^{x^{\prime}} k(x, x^{\prime})
\end{align}
ただし$\calA_p^x, \calA_p^{x^{\prime}}$はそれぞれ変数$x, x^{\prime}$についてのStein operatorであり，$\kappa_p (x, x^{\prime})$は$k(x, x^{\prime})$にStein operatorを2回適用した"Steinalized" kernel．

---

#### Proof of Equation (1)

\begin{align}
\mathbb{E}\_{x \sim p} \left[f(x) \nabla_x \log p(x) + \nabla_x f(x) \right]
&= \int_{-\infty}^{\infty} p(x) \left\\{ f(x) \nabla_x \log p(x) + \nabla_x f(x) \right\\} dx \newline
&= \int_{-\infty}^{\infty} p(x) \left\\{f(x) \frac{\nabla_x p(x)}{p(x)} + \nabla_x f(x) \right\\} dx \newline
&= \int_{-\infty}^{\infty} \left\\{f(x) \nabla_x p(x) + p(x) \nabla_x f(x) \right\\} dx \newline
&= \left[p(x) f(x) \right]_{-\infty}^{\infty} = 0 - 0 = 0
\end{align}
