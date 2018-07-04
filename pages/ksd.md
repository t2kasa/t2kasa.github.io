---
title: Kernelized Stein Discrepancy
tags: [Work In Progress]
sidebar: home_sidebar
permalink: ksd.html
---

[A Short Introduction to Kernelized Stein Discrepancy](http://www.cs.dartmouth.edu/~qliu/PDF/ksd_short.pdf)  
Kernelized Stein Discrepancyについて現状わかっている範囲でメモ。

### Stein's Identity

微分可能な確率密度関数$p(x)$と関数$f(x) \in \mathbb{R}$があり，$\lim_{\parallel x \parallel \rightarrow \infty} p(x)f(x) = 0$のとき

\begin{align}
\mathbb{E}_{x \sim p} \left[f(x) \nabla_x \log p(x) + \nabla_x f(x) \right] = 0 \hspace{10pt} \forall f
\end{align}

を満たす．ここから$\mathbb{E}_{x \sim p} \left[g(x;p) \right] = 0$を満たす$g(x;p)$について

\begin{align}
\mathbb{E}_{x \sim p} \left[f(x) \nabla_x \log p(x) + \nabla_x f(x) \right] = \mathbb{E}\_{x \sim p} \left[g(x;p) \right]
\end{align}

という等式が得られる．これは*Stein equation*と呼ばれるらしい．
これを解くと

\begin{align}
f(x) = \frac{1}{p(x)} \int_a^x g(\xi;p) p(\xi) d\xi
\end{align}

が得られる．

#### 式(1)の証明

\begin{align}
\mathrm{lhs} &= \int_{-\infty}^{\infty} p(x) \left\\{ f(x) \nabla_x \log p(x) + \nabla_x f(x) \right\\} dx \newline
&= \int_{-\infty}^{\infty} p(x) \left\\{f(x) \frac{\nabla_x p(x)}{p(x)} + \nabla_x f(x) \right\\} dx \newline
&= \int_{-\infty}^{\infty} \left\\{f(x) \nabla_x p(x) + p(x) \nabla_x f(x) \right\\} dx \newline
&= \left[p(x) f(x) \right]_{-\infty}^{\infty} = 0 - 0 = 0
\end{align}

#### 式(2)の証明