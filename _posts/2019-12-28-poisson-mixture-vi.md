---
title: "ポアソン混合分布で変分推論"
permalink: 2019-12-28-poisson-mixture-vi.html
sidebar: blog_sidebar
---

\begin{align}
\newcommand{\b}[1]{\mathbf{#1}}
\newcommand{\bo}[1]{\boldsymbol{#1}} \nonumber
\end{align}

[ベイズ推論による機械学習入門][bayes-book]より，ポアソン混合分布で変分推論をしてみます．
記号などは[前回の記事](https://t2kasa.github.io/2019-12-27-poisson-mixture-gibbs.html)と同様になっているので，まずはそちらを確認してください．

事後分布を潜在変数の確率分布とパラメータの確率分布に分解できると仮定して近似します．
\begin{align}
p(\b{S}, \bo{\lambda}, \bo{\pi} | \b{X}) \approx q(\b{S}) q(\bo{\lambda}, \bo{\pi})
\end{align}

まず$q(\b{S})$について考えます．
\begin{align}
\ln q(\b{S}) 
&= \left< \ln p(\b{X}, \b{S}, \bo{\lambda}, \bo{\pi}) \right>\_{q(\bo{\lambda}, \bo{\pi})} + \mathrm{const.} \newline
&= \left< \ln p(\b{X} | \b{S}, \bo{\lambda}) + \ln p(\b{S} | \bo{\pi}) + \ln p(\bo{\lambda}) + \ln p(\bo{\pi}) \right>\_{q(\bo{\lambda}, \bo{\pi})} + \mathrm{const.} \newline
&= \left< \ln p(\b{X} | \b{S}, \bo{\lambda}) \right>\_{q(\bo{\lambda})} + \left< \ln p(\b{S} | \bo{\pi}) \right>\_{q(\bo{\pi})} + \mathrm{const.} \newline
&= \sum_{n = 1}^N \left\\{ \left< \ln p(x_n | \b{s}_n, \bo{\lambda}) \right>\_{q(\bo{\lambda})} + \left< \ln p(\b{s}_n | \bo{\pi}) \right>\_{q(\bo{\pi})} \right\\} + \mathrm{const.}
\end{align}
各項について考えます．
\begin{align}
\left< \ln p(x_n | \b{s}_n, \bo{\lambda}) \right>\_{q(\bo{\lambda})} &= \sum\_{k = 1}^K \left< s\_{n, k} \ln \mathrm{Poi} (x_n | \lambda_k) \right>\_{q(\lambda_k)}
= \sum\_{k = 1}^K s\_{n, k} \left( x_n \left< \ln \lambda_k \right> - \left< \lambda_k \right> \right) + \mathrm{const.} \newline
\left< \ln p(\b{s}_n | \bo{\pi}) \right>\_{q(\bo{\pi})} &= \sum\_{k = 1}^K \left< \ln \mathrm{Cat} (\b{s}_n | \bo{\pi}) \right>\_{q(\lambda_k)} = \sum\_{k = 1}^K s\_{n, k} \left< \ln \pi_k \right>
\end{align}
よって$\ln q(\b{S})$は
\begin{align}
\ln q(\b{S}) = \sum\_{n = 1}^N \left\\{ \sum\_{k = 1}^K s\_{n, k} \left( x_n \left< \ln \lambda_k \right> - \left< \lambda_k \right> + \left< \ln \pi_k \right> \right) \right\\} + \mathrm{const.}
\end{align}
となります．したがって
\begin{align}
q(\b{s}_n) = \mathrm{Cat} (\b{s}_n | \bo{\eta}_n)
\end{align}
ただし
\begin{align}
\eta\_{n, k} \propto \exp \left\\{ x_n \left< \ln \lambda_k \right> - \left< \lambda_k \right> + \left< \ln \pi_k \right> \right\\} \quad \left( \mathrm{s.t.} \; \sum\_{k = 1}^K \eta\_{n, k} = 1 \right)
\end{align}

次に，$q(\bo{\lambda}, \bo{\pi})$について考えます．
\begin{align}
\ln q(\bo{\lambda}, \bo{\pi}) 
&= \left< \ln p(\b{X}, \b{S}, \bo{\lambda}, \bo{\pi}) \right>\_{q(\b{S})} + \mathrm{const.} \newline
&= \left< \ln p(\b{X} | \b{S}, \bo{\lambda}) + \ln p(\b{S} | \bo{\pi}) + \ln p(\bo{\lambda}) + \ln p(\bo{\pi}) \right>\_{q(\b{S})} + \mathrm{const.} \newline
&= \left< \ln p(\b{X} | \b{S}, \bo{\lambda}) \right>\_{q(\b{S})} + \left< \ln p(\b{S} | \bo{\pi}) \right>\_{q(\b{S})} + \ln p(\bo{\lambda}) + \ln p(\bo{\pi}) + \mathrm{const.} \newline
\end{align}

$\bo{\lambda}$と$\bo{\pi}$に関する項が分解されているので，$q(\bo{\lambda}, \bo{\pi}) = q(\bo{\lambda}) q(\bo{\pi})$に分解できることが分かります．

\begin{align}
\ln q(\bo{\lambda}) 
&= \left< \ln p(\b{X} | \b{S}, \bo{\lambda}) \right>\_{q(\b{S})} + \ln p(\bo{\lambda}) + \mathrm{const.} \newline
&= \sum_{n = 1}^N \left< \ln p(x_n | \b{s}_n, \bo{\lambda}) \right>\_{q(\b{s}_n)} + \sum\_{k = 1}^K \ln \mathrm{Gam} (\lambda_k | a, b) + \mathrm{const.} \newline
&= \sum\_{n = 1}^N \left< \sum\_{k = 1}^K s\_{n, k} \ln \mathrm{Poi} (x_n | \lambda_k) \right>\_{q(\b{s}_n)} + \sum\_{k = 1}^K \ln \mathrm{Gam} (\lambda_k | a, b) + \mathrm{const.} \newline
&= \sum\_{n = 1}^N \sum\_{k = 1}^K \left< s\_{n, k} \right> (x_n \ln \lambda_k - \lambda_k) + \sum\_{k = 1}^K \left\\{ (a - 1) \ln \lambda_k - b \lambda_k \right\\} + \mathrm{const.} \newline
&= \sum\_{k = 1}^K \left\\{ \left( \sum\_{n = 1}^N \left< s\_{n, k} \right> x_n + a - 1 \right) \ln \lambda_k - \left( \sum\_{n = 1}^N \left< s\_{n, k} \right> + b \right) \lambda_k \right\\} + \mathrm{const.}
\end{align}

よって
\begin{align}
q(\lambda_k) = \mathrm{Gam} ( \lambda_k | \hat{a}\_k, \hat{b}\_k)
\end{align}
\begin{align}
\hat{a}\_k = \sum\_{n = 1}^N \left< s_{n, k} \right> x_n + a, \; \hat{b}\_k = \sum\_{n = 1}^N \left< s_{n, k} \right> + b
\end{align}

次は$q(\bo{\pi})$です．

\begin{align}
\ln q(\bo{\pi}) 
&= \left< \ln p(\b{S} | \bo{\pi}) \right>\_{q(\b{S})} + \ln p(\bo{\pi}) + \mathrm{const.} \newline
&= \sum_{n = 1}^N \left< \ln p(\b{s}_n | \bo{\pi}) \right>\_{q(\b{s}_n)} + \ln \mathrm{Dir} (\bo{\pi} | \bo{\alpha}) + \mathrm{const.} \newline
&= \sum\_{n = 1}^N \left< \ln \mathrm{Cat} (\b{s}_n | \bo{\pi}) \right>\_{q(\b{s}_n)} + \ln \mathrm{Dir} (\bo{\pi} | \bo{\alpha}) + \mathrm{const.} \newline
&= \sum\_{n = 1}^N \left< \sum\_{k = 1}^K s\_{n, k} \ln \pi_k \right>\_{q(\b{s}_n)} + \sum\_{k = 1}^K (\alpha_k - 1) \ln \pi_k + \mathrm{const.} \newline
&= \sum\_{k = 1}^K \left\\{ \left( \sum\_{n = 1}^N \left< s\_{n, k} \right> + \alpha_k - 1 \right) \ln \pi_k \right\\} + \mathrm{const.}
\end{align}

よって
\begin{align}
q(\bo{\pi}) = \mathrm{Dir} (\bo{\pi} | \hat{\bo{\alpha}})
\end{align}
\begin{align}
\hat{\alpha}\_k = \sum\_{n = 1}^N \left< s\_{n, k} \right> + \alpha_k
\end{align}

以上より，各確率分布が得られたので，対応する期待値も求められます．
- $ q(\b{s}\_n) = \mathrm{Cat} (\b{s}\_n \| \bo{\eta}\_n ) $ より $ \left< s_{n, k} \right> = \eta_{n, k} $
- $ q(\lambda_k) = \mathrm{Gam} (\lambda_k \| \hat{a}_k, \hat{b}_k) $ より $ \left< \lambda_k \right> = \hat{a}_k / \hat{b}_k, \; \left< \ln \lambda_k \right> = \psi (\hat{a}_k) - \ln \hat{b}_k $
- $q(\bo{\pi}) = \mathrm{Dir} (\bo{\pi} \| \hat{\bo{\alpha}})$ より $\left< \ln \pi_k \right> = \psi(\hat{\alpha}_k) - \psi(\sum\_{i = 1}^K \hat{\alpha}_i )$

ここで$\psi(\cdot)$はディガンマ関数です．

## 実装

<script src="https://gist.github.com/t2kasa/e5eccd33c604290d8801b42ab38128d2.js"></script>

## References

- [ベイズ推論による機械学習入門](https://www.kspub.co.jp/book/detail/1538320.html)

[bayes-book]: https://www.kspub.co.jp/book/detail/1538320.html
