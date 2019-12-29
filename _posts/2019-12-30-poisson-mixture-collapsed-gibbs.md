---
title: "ポアソン混合分布で崩壊型ギブスサンプリング"
permalink: 2019-12-30-poisson-mixture-collapsed-gibbs.html
sidebar: blog_sidebar
---

\begin{align}
\newcommand{\b}[1]{\mathbf{#1}}
\newcommand{\bo}[1]{\boldsymbol{#1}} \nonumber
\end{align}

[ベイズ推論による機械学習入門][bayes-book]より，ポアソン混合分布で崩壊型ギブスサンプリングをしてみます．
記号などは[以前の記事](https://t2kasa.github.io/2019-12-27-poisson-mixture-gibbs.html)と同様になっているので，まずはそちらを確認してください．

崩壊型ギブスサンプリングの考え方やポアソン混合分布に対する適用は
[ベイズ推論による機械学習入門][bayes-book]の[著者のブログ記事][collapsed-gibbs]にも解説がありますので，そちらもご覧ください．

さて，同時分布からパラメータを周辺化し，周辺化後の確率分布から$\b{S}$をサンプリングすることを考えます．
\begin{align}
p(\b{X}, \b{S}) = \int \int p(\b{X}, \b{S}, \bo{\lambda}, \bo{\pi}) d\bo{\lambda} d\bo{\pi}
\end{align}
具体的には，事後分布$p(\b{S} | \b{X})$に対してギブスサンプリングを適用します．
$\b{S}$から$\b{s}\_n$を除いた$\b{S}\_{\backslash n}$が与えられたとして$\b{s}_n$をサンプリングすることを考えます．

\begin{align}
p(\b{s}_n | \b{X}, \b{S}\_{\backslash n}) 
&\propto p(x_n, \b{X}\_{\backslash n}, \b{s}_n, \b{S}\_{\backslash n}) \newline
&= p(x_n | \b{X}\_{\backslash n}, \b{s}\_n, \b{S}\_{\backslash n}) p(\b{X}\_{\backslash n} | \b{s}\_n, \b{S}\_{\backslash n}) p(\b{s}\_n | \b{S}\_{\backslash n}) p(\b{S}\_{\backslash n}) \newline
&\propto p(x_n | \b{X}\_{\backslash n}, \b{s}\_n, \b{S}\_{\backslash n}) p(\b{s}\_n | \b{S}\_{\backslash n}) \newline
\end{align}

上述の式変形後の確率分布が$s_{n, k} = 1$となる場合の確率をそれぞれ求め，正規化することによって$\b{s}_n$をサンプリングするカテゴリカル分布のパラメータとすることができます．即ち，

\begin{align}
p(\b{s}\_n | \b{X}, \b{S}\_{\backslash n}) = \mathrm{Cat} ( \b{s}_n | \bo{\zeta}_n)
\end{align}

\begin{align}
\zeta_{n, k} = \frac{p(x_n | \b{X}\_{\backslash n}, s_{n, k} = 1, \b{S}\_{\backslash n}) p( s_{n, k} = 1 | \b{S}\_{\backslash n})}{\sum\_{i = 1}^K p(x_n | \b{X}\_{\backslash n}, s_{n, i} = 1, \b{S}\_{\backslash n}) p( s_{n, i} = 1 | \b{S}\_{\backslash n})}
\end{align}

とします．

これらの確率分布の導出過程は省略します（書籍から確認ください）．事後分布と予測分布の導出を用いることになります．
\begin{align}
p(\b{s}\_n | \b{S}\_{\backslash n}) = \mathrm{Cat} (\b{s}\_n | \bo{\eta}\_{\backslash n})
\end{align}
ただし
\begin{align}
\eta\_{\backslash n, k} \propto \hat{\alpha}\_{\backslash n, k}
\end{align}
\begin{align}
\hat{\alpha}\_{\backslash n, k} = \sum\_{n' \neq n} s\_{n', k} + \alpha_k
\end{align}

\begin{align}
p(x_n | \b{X}\_{\backslash n}, s_{n, k} = 1, \b{S}\_{\backslash n}) = \mathrm{NB} \left( x\_n | \hat{a}\_{\backslash n, k}, \frac{1}{\hat{b}\_{\backslash n, k} + 1} \right)
\end{align}
ただし
\begin{align}
\hat{a}\_{\backslash n, k} &= \sum\_{n' \neq n} s\_{n', k} x\_{n'} + a \newline
\hat{b}\_{\backslash n, k} &= \sum\_{n' \neq n} s\_{n', k} + b
\end{align}

## 実装

Pythonで実装しました．負の二項分布でscipyなどの実装を用いる場合は$p$と$1 - p$が書籍と逆になっている点に注意が必要です．

書籍中で記載されているパラメータの効率的な更新方法を用いる場合と用いない場合の両方を示しています．
10回ぐらいの$\b{S}$のサンプリングでも十分に良いサンプリング結果が得られているようです．

<script src="https://gist.github.com/t2kasa/aeda14bb18cdc667a0dca951b92f0782.js"></script>

## References

- [ベイズ推論による機械学習入門](https://www.kspub.co.jp/book/detail/1538320.html)
- [ベイズ混合モデルにおける近似推論③ ～崩壊型ギブスサンプリング～](http://machine-learning.hatenablog.com/entry/2016/11/03/205317)

[bayes-book]: https://www.kspub.co.jp/book/detail/1538320.html
[collapsed-gibbs]: http://machine-learning.hatenablog.com/entry/2016/11/03/205317