---
title: "ポアソン混合分布でギブスサンプリング"
permalink: 2019-12-27-poisson-mixture-gibbs.html
sidebar: blog_sidebar
---

\begin{align}
\newcommand{\b}[1]{\mathbf{#1}}
\newcommand{\bo}[1]{\boldsymbol{#1}} \nonumber
\end{align}

[ベイズ推論による機械学習入門][bayes-book]より，ポアソン混合分布でギブスサンプリングをしてみます．記号の表記は書籍に合わせます．考え方の詳細は書籍を参照してください．

記号の表記は以下の通りです．

- クラスタの数$K$
- クラスタの混合比率$\bo{\pi}$
- クラスタの割り当て$\b{s}_n$
- クラスタ$k$のパラメータ$\bo{\theta}_k$

$N$個のデータ$\b{X} = \left\\{ \b{x}_1, \ldots, \b{x}_N \right\\}$が観測されたときの同時分布を次のように設計します．
\begin{align}
p(\b{X}, \b{S}, \bo{\Theta}, \bo{\pi}) 
&= p(\b{X} | \b{S}, \bo{\Theta}) p(\b{S} | \bo{\pi}) p(\bo{\Theta}) p(\bo{\pi}) \newline
&= \left\\{ \prod\_{n = 1}^N p(\b{x}_n | \b{s}_n, \bo{\Theta}) p(\b{s}_n | \bo{\pi}) \right\\} \left\\{ \prod\_{n = 1}^N p(\bo{\theta}_k) \right\\} p(\bo{\pi})
\end{align}

ここで
\begin{align}
p(\b{x}_n | \b{s}_n, \bo{\Theta}) &= \prod\_{k = 1}^K p(\b{x}_n | \bo{\theta}_k)^{s\_{n, k}} \newline
p(\b{s}_n | \bo{\pi}) &= \mathrm{Cat} (\b{s}_n | \bo{\pi}) = \prod\_{k = 1}^K \pi_k^{s\_{n, k}} \newline
p(\bo{\pi}) &= \mathrm{Dir} (\bo{\pi} | \bo{\alpha}) = C\_{\mathrm{D}} (\bo{\alpha}) \prod\_{k = 1}^K \pi_k^{\alpha_k - 1}
\end{align}
です．

## ポアソン混合分布におけるギブスサンプリング

ポアソン混合分布ではクラスタ$k$の観測モデルを
\begin{align}
p(x_n | \lambda_k) = \mathrm{Poi} (x_n | \lambda_k) = \frac{\lambda_k^{x_n}}{x_n !} e^{-\lambda_k}
\end{align}
とします．また，ポアソン分布のパラメータ$\bo{\lambda} = \left\\{\lambda_1, \ldots, \lambda_K \right\\}$に対する事前分布として共役事前分布のガンマ分布を用います．
\begin{align}
p(\lambda_k) = \mathrm{Gam} (\lambda_k | a, b) = C\_{\mathrm{G}}(a, b) \lambda_k^{a - 1} e^{-b \lambda_k}
\end{align}

さて，ここからが本題です．$\b{X}$が観測されたとして，ギブスサンプリングによって他のパラメータをサンプリングします．
\begin{align}
\b{S} &\sim p(\b{S} | \b{X}, \bo{\lambda}, \bo{\pi}) \newline
\bo{\lambda}, \bo{\pi} &\sim p(\bo{\lambda}, \bo{\pi} | \b{X}, \b{S})
\end{align}

まず$\b{S}$にのみ注目して，$p(\b{S} \| \b{X}, \bo{\lambda}, \bo{\pi})$を求めます．
\begin{align}
p(\b{S} | \b{X}, \bo{\lambda}, \bo{\pi}) 
&= \frac{p(\b{X}, \b{S}, \bo{\lambda}, \bo{\pi})}{p(\b{X}, \bo{\lambda}, \bo{\pi})} \propto p(\b{X}, \b{S}, \bo{\lambda}, \bo{\pi}) = p(\b{X} | \b{S}, \bo{\lambda}) p(\b{S} | \bo{\pi}) p(\bo{\lambda}) p(\bo{\pi}) \newline
&\propto p(\b{X} | \b{S}, \bo{\lambda}) p(\b{S} | \bo{\pi}) = \prod\_{n = 1}^N p(x_n | \b{s}_n, \bo{\lambda}) p(\b{s}_n | \bo{\pi})
\end{align}

対数をとると

\begin{align}
\ln p(x_n | \b{s}\_n, \bo{\lambda}) 
&= \sum\_{k = 1}^K s_{n, k} \ln \mathrm{Poi} (x_n | \lambda_k)
= \sum_{k = 1}^K s_{n, k} \left( x_n \ln \lambda_k - \ln x_n ! - \lambda_k \right) \newline
&= \sum_{k = 1}^K s_{n, k} \left( x_n \ln \lambda_k - \lambda_k \right) - \ln x_n ! \quad \left( \because \sum_{k = 1}^K s_{n, k} \ln x_n ! = \ln x_n ! \right) \newline
&= \sum_{k = 1}^K s_{n, k} \left( x_n \ln \lambda_k - \lambda_k \right) + \mathrm{const.}
\end{align}
となります．途中の式変形でクラスタの割り当てについて$\sum_{k = 1}^K s_{n, k} = 1$となることを用いました．また

\begin{align}
\ln p(\b{s}\_n | \bo{\pi}) = \ln \mathrm{Cat} (\b{s}\_n | \bo{\pi}) = \sum\_{k = 1}^K s_{n, k} \ln \pi_k
\end{align}
です．よって

\begin{align}
\ln p(x_n | \b{s}\_n, \bo{\lambda}) p(\b{s}\_n | \bo{\pi}) = \sum_{k = 1}^K s_{n, k} \left( x_n \ln \lambda_k - \lambda_k + \ln \pi_k \right) + \mathrm{const.}
\end{align}
を得ます．これはカテゴリ分布に対数をとったものと考えられるので，
\begin{align}
\b{s}\_n &\sim \mathrm{Cat} (\b{s}_n | \bo{\eta}) \newline
\eta\_{n, k} &\propto \exp \left\\{ x_n \ln \lambda_k - \lambda_k + \ln \pi_k \right\\} \quad \left( \mathrm{s.t.} \sum\_{k = 1}^K \eta\_{n, k} = 1 \right)
\end{align}
となります．

次に，$\bo{\lambda}, \bo{\pi}$に注目します．
\begin{align}
p(\bo{\lambda}, \bo{\pi} | \b{X}, \b{S})
&= \frac{p(\b{X}, \b{S}, \bo{\lambda}, \bo{\pi})}{p(\b{X}, \b{S})} \propto p(\b{X}, \b{S}, \bo{\lambda}, \bo{\pi}) = p(\b{X} | \b{S}, \bo{\lambda}) p(\b{S} | \bo{\pi}) p(\bo{\lambda}) p(\bo{\pi}) \newline
&= \left\\{ p(\b{X} | \b{S}, \bo{\lambda}) p(\bo{\lambda}) \right\\} \left\\{ p(\b{S} | \bo{\pi}) p(\bo{\pi}) \right\\}
\end{align}
$\bo{\lambda}, \bo{\pi}$の確率分布は上述のように分解できることが分かるので，各パラメータについて別々に考えることができます．まず$\bo{\lambda}$について考えます．
\begin{align}
\ln p(\b{X} | \b{S}, \bo{\lambda}) p(\bo{\lambda})
&= \sum_{n = 1}^N \ln p(x_n | \b{s}\_n, \bo{\lambda}) + \sum\_{k = 1}^K \ln \mathrm{Gam} (\lambda_k | a, b) \newline
&= \sum_{n = 1}^N \sum_{k = 1}^K s_{n, k} \ln \mathrm{Poi} (x_n | \lambda_k) + \sum\_{k = 1}^K \ln \mathrm{Gam} (\lambda_k | a, b) \newline
&= \sum_{n = 1}^N \sum_{k = 1}^K s_{n, k} \left( x_n \ln \lambda_k - \underbrace{\ln x_n !}\_{\mathrm{const.}} - \lambda_k \right) + \sum_{k = 1}^K \left\\{ (a - 1) \ln \lambda_k - b \lambda_k + \underbrace{\ln C_{\mathrm{G}} (a, b)}_{\mathrm{const.}} \right\\} \newline
&= \sum\_{n = 1}^N \sum\_{k = 1}^K s\_{n, k} \left( x_n \ln \lambda_k - \lambda_k \right) + \sum\_{k = 1}^K \left\\{ (a - 1) \ln \lambda_k - b \lambda_k \right\\} + \mathrm{const.} \newline
&= \sum\_{k = 1}^K \left\\{ \left( \sum\_{n = 1}^N s\_{n, k} x_n + a - 1 \right) \ln \lambda_k - \left( \sum\_{n = 1}^N s\_{n, k} + b \right) \lambda_k \right\\} + \mathrm{const.}
\end{align}
よって，$\lambda_k$の確率分布はガンマ分布になります．
\begin{align}
\lambda_k &\sim \mathrm{Gam} (\lambda_k | \hat{a}_k, \hat{b}_k) \newline
\hat{a}_k &= \sum\_{n = 1}^N s\_{n, k} x_n + a, \hat{b}_k = \sum\_{n = 1}^N s\_{n, k} + b
\end{align}

次に$\bo{\pi}$について考えます．
\begin{align}
\ln p(\b{S} | \bo{\pi}) p(\bo{\pi})
&= \sum_{n = 1}^N \ln \mathrm{Cat} (\b{s}_n | \bo{\pi}) + \ln \mathrm{Dir} (\bo{\pi} | \bo{\alpha}) \newline
&= \sum\_{n = 1}^N \sum\_{k = 1}^K s\_{n, k} \ln \pi_k + \sum\_{k = 1}^K (\alpha_k - 1) \ln \pi_k + \mathrm{const.} \newline
&= \sum\_{k = 1}^K \left( \sum\_{n = 1}^N s\_{n, k} + \alpha_k - 1 \right) \ln \pi_k + \mathrm{const.}
\end{align}
よって$\bo{\pi}$の確率分布は以下のディリクレ分布になります．
\begin{align}
\bo{\pi} &\sim \mathrm{Dir} (\bo{\pi} | \hat{\bo{\alpha}}) \newline
\hat{\alpha}_k &= \sum\_{n = 1}^N s\_{n, k} + \alpha_k
\end{align}

## 実装

$\eta_{n, k}$を算出する際に，そのまま指数計算を行うとオーバーフローしてしまう可能性があります．そこでlog-sum-expを用いることで，これを回避します．
$ \ln \zeta_{n, k} = x_n \ln \lambda_k - \lambda_k + \ln \pi_k$とすると
\begin{align}
\eta\_{n, j} = \frac{ \exp \left\( \ln \zeta_{n, j} \right) }{\sum_{k = 1}^K \exp \left( \ln \zeta_{n, k} \right)}
\end{align}
として$\eta_{n, k}$を求めることになります．ここで対数をとることで
\begin{align}
\ln \eta\_{n, j} = \ln \frac{ \exp \left\( \ln \zeta_{n, j} \right) }{\sum_{k = 1}^K \exp \left( \ln \zeta_{n, k} \right)} = \ln \zeta_{n, j} - \ln \sum_{k = 1}^K \exp \left( \ln \zeta_{n, k} \right)
\end{align}
となり，log-sum-expを用いることで，安定して求めることができます．

<script src="https://gist.github.com/t2kasa/2825e831fa06185dec219c3c6cfe4f42.js"></script>

## References

- [ベイズ推論による機械学習入門](https://www.kspub.co.jp/book/detail/1538320.html)
- [「ベイズ推論による機械学習入門」を読んだので実験してみた(その2)](http://szdr.hatenablog.com/entry/2017/12/10/025054)

[bayes-book]: https://www.kspub.co.jp/book/detail/1538320.html
