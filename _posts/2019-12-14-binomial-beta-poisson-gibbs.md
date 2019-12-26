---
title: "Binomial-Beta-Poissonでギブスサンプリング"
permalink: 2019-12-14-binomial-beta-poisson-gibbs.html
sidebar: blog_sidebar
---

\begin{align}
\newcommand{\b}[1]{\mathbf{#1}}
\newcommand{\bo}[1]{\boldsymbol{#1}} \nonumber
\end{align}

[こちらのスライド][gibbs]{:target="_blank"}でexampleとして挙げられているBinomial-Beta-Poissonでギブスサンプリングをやってみます．

## Binomial-Beta-Poissonでのギブスサンプリングアルゴリズム

各確率変数に対する確率分布を以下のように考えます．ここで，$\lambda, a, b$は既知とします．

\begin{align}
p(n | \lambda) &= \mathrm{Poi} (n | \lambda) = e^{-\lambda} \frac{\lambda^n}{n!} \newline
p(\theta | a, b) &= \mathrm{Beta} (\theta | a, b) = C_{\mathrm{B}} (a, b) \theta^{a - 1} (1 - \theta)^{b - 1} \newline
p(x | n, \theta) &= \mathrm{Bin} (x |n, \theta) = \binom{n}{x} \theta^x (1 - \theta)^{n - x}
\end{align}

ここで，結合分布$p(x, \theta, n)$が以下のように分解できると仮定します．
\begin{align}
p(x, \theta, n) = p(x | n, \theta) p(n) p(\theta) \propto \binom{n}{x} \theta^{a + x - 1} (1 - \theta)^{n + b - x - 1} \frac{\lambda^n}{n!}
\end{align}

ギブスサンプリングのために，各確率変数について他の確率変数で条件付けた確率分布を求めます．
まず，$p(x | n, \theta)$については上述の通り，
\begin{align}
p(x | n, \theta) = \binom{n}{x} \theta^x (1 - \theta)^{n - x} \propto \mathrm{Bin} (x |n, \theta)
\end{align}
です．次に$p(\theta | x, n)$を考えます．$\theta$に依存しない項は定数とみなすことで次のように変形できます．
\begin{align}
p(\theta | x, n)
&= \frac{p(x, \theta, n)}{p(x, n)} = \frac{p(x | n, \theta) p(\theta) p(n)}{ p(x, n)} \propto p(x | n, \theta) p(\theta) \newline
&= \binom{n}{x} \theta^x (1 - \theta)^{n - x} \cdot C_{\mathrm{B}} (a, b) \theta^{a - 1} (1 - \theta)^{b - 1} \newline
&\propto \theta^{a + x - 1} (1 - \theta)^{n + b - x - 1} \propto \mathrm{Beta} (\theta | a + x, n + b - x)
\end{align}
最後に$p(n | \theta, x)$を考えます．$n$に依存しない項は定数とみなすと
\begin{align}
p(n | \theta, x) 
&= \frac{p(x, \theta, n)}{p(\theta, x)} = \frac{p(x | n, \theta) p(\theta) p(n)}{p(\theta, x)} \propto p(x | n, \theta) p(n) \newline
&= \binom{n}{x} \theta^x (1 - \theta)^{n - x} \cdot e^{-\lambda} \frac{\lambda^n}{n!} \newline
&= \frac{n!}{(n - x)! x!} \theta^x (1 - \theta)^{n - x} \cdot e^{-\lambda} \frac{\lambda^n}{n!} \newline
&= \frac{1}{(n - x)! x!} \theta^x (1 - \theta)^{n - x} \cdot e^{-\lambda} \lambda^n \newline
&\propto \frac{(1 - \theta)^{n - x} \lambda^n}{(n - x)!} \newline
&\propto \frac{(1 - \theta)^{n - x} \lambda^{n - x}}{(n - x)!} \newline
&= \frac{ \left\\{ \lambda (1 - \theta) \right\\}^{n - x}}{(n - x)!}
\end{align}
となります．ここで$z = n - x$とおくと
\begin{align}
\frac{ \left\\{ \lambda (1 - \theta) \right\\}^{n - x}}{(n - x)!} = \frac{ \left\\{ \lambda (1 - \theta) \right\\}^z}{z!} \propto \mathrm{Poi} (z | \lambda (1 - \theta))
\end{align}
が得られます．

よってBinomial-Beta-Poissonにおけるギブスサンプリングのアルゴリズムは以下のようになります．

<div style="border-bottom: 1px solid gray;"></div>

- Set initial values ($x^{(0)}$, $\theta^{(0)}$, $n^{(0)}$)
- for $i = 1, 2, \ldots, N$ do
    - Sample $x^{(i)} \sim \mathrm{Bin} (x \|n^{(i - 1)}, \theta^{(i - 1)})$
    - Sample $\theta^{(i)} \sim \mathrm{Beta} (\theta \| a + x^{(i)}, n^{(i - 1)} + b - x^{(i)})$
    - Sample $n^{(i)} = x^{(i)} + z, z \sim \mathrm{Poi} (z \| \lambda (1 - \theta^{(i)}))$
- end for

<div style="border-bottom: 1px solid gray;"></div>

## 実装

<script src="https://gist.github.com/t2kasa/ae04ac0c44df891c598f5014abddd96e.js"></script>

## References

- [Some Examples on Gibbs Sampling and Metropolis-Hastings methods](http://www.stat.unm.edu/~ghuerta/stat574/notes-gibbs-metro.pdf){:target="_blank"}
- [ベイズ推論による機械学習入門](https://www.kspub.co.jp/book/detail/1538320.html)

[gibbs]: http://www.stat.unm.edu/~ghuerta/stat574/notes-gibbs-metro.pdf
