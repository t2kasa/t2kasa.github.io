---
title:  "モンテカルロフィルタを試す (Monte Carlo Filter)"
permalink: 2018-11-11-monte-carlo-filter.html
sidebar: blog_sidebar
tags: []
---

\begin{align}
\newcommand{\bx}{\mathbf{x}}
\newcommand{\by}{\mathbf{y}}
\newcommand{\bv}{\mathbf{v}}
\newcommand{\bw}{\mathbf{w}}
\newcommand{\bSigma}{\mathbf{\Sigma}} \nonumber
\end{align}

唐突だが，パーティクルフィルタに入門しようと思った．
[21世紀の統計科学 Vol. III 数理・計算の統計科学](http://park.itc.u-tokyo.ac.jp/atstat/jss75shunen/Vol3.pdf)
の10章・11章あたりを読んでいたが，結構分かりやすかった．
勉強の一区切りとして，パーティクルフィルタの中でもシンプルな特殊系であるモンテカルロフィルタを実装してみることにした．

時刻$k$において，状態を表す変数を$\bx_k$とし，観測値を表す変数を$\by_k$とする．
状態の遷移を表すシステムモデルを$f(\bx_k|\bx_{k - 1})$，状態から観測値を得る観測モデルを$h(\by_k|\bx_k)$とする．いずれのモデルも確率分布として表現される．
パーティクルフィルタはフィルタリング確率分布$p(\bx_k|\by_{1:k})$を$M$個の重み付き粒子によって近似する．

\begin{align}
    \left\\{\left(\bx_k^{(i)}, w_k^{(i)} \right) \right\\}_{i = 1}^M \nonumber
\end{align}

モンテカルロフィルタは提案分布としてシステムモデル
$f(\bx_k | \bx_{k - 1})$
を用いる特殊な場合になっている．

## アルゴリズム

モンテカルロフィルタのアルゴリズムは以下の通り（通常のパーティクルフィルタのアルゴリズムの詳細は参考文献参照）．
<div style="border-bottom: 1px solid gray;"></div>
アルゴリズム：モンテカルロフィルタ
<div style="border-bottom: 1px solid gray;"></div>

入力：
$\left\\{\bx_{k - 1}^{(i)}_{i = 1}^M \right\\} \sim p(\bx_{x - 1} | \by_{1:k - 1})$  
出力：
$\left\\{\bx_k^{(i)} \right\\}_{i = 1}^M \sim p(\bx_k | \by_{1:k})$  

(A1) システムモデルによって時刻$k - 1$の粒子群$\left\\{ \bx_{k - 1}^{(i)} \right\\}\_{i = 1}^M$から時刻$k$の粒子群$\left\\{ \bx_k^{(i)} \right\\}\_{i = 1}^M$を生成．

\begin{align}
    \tilde{\bx}\_k^{(i)} \sim f( \bx_k|\bx_{k - 1}^{(i)} )
\end{align}

(A2) 時刻$k$の観測値$\by_k$と観測モデルを用いて各粒子の尤度，および尤度に比例する重みを算出．

\begin{align}
    \tilde{w}_k^{(i)} \propto h(\by_k|\tilde{\bx}_k^{(i)})
\end{align}

(A3) 重みを正規化．

\begin{align}
    w_k^{(i)} = \frac{\tilde{w}\_k^{(i)}}{\sum_{i = 1}^M \tilde{w}_k^{(i)}}
\end{align}

(A4) 重みに比例する確率で粒子群を復元抽出．

\begin{align}
    \bx_k^{(i)} \sim
    \left\\{ \begin{array}{lll}
        \tilde{\bx}_k^{(1)} & \mathrm{with \; prob.} & w_k^{(1)} \newline
        \tilde{\bx}_k^{(2)} & \mathrm{with \; prob.} & w_k^{(2)} \newline
        \vdots & & \vdots \newline
        \tilde{\bx}_k^{(M)} & \mathrm{with \; prob.} & w_k^{(M)}
    \end{array} \right.
\end{align}

<div style="border-bottom: 1px solid gray;"></div>

## 実装

非常にシンプルなケースを実装してみる。
システムモデル・観測モデルはともに1次元のランダムウォークとする．
また，ノイズは時刻によらず同一の世紀分布に従うとする．
即ち，

\begin{align}
    \bx\_k &= \bx\_{k - 1} + \bv, \quad \bv \sim \mathcal{N}(\mathbf{0}, \bSigma_v) \newline
    \by_k &= \bx_k + \bw, \quad \bw \sim \mathcal{N}(\mathbf{0}, \bSigma_w)
\end{align}

である．全てベクトル・行列表記にしているが実装上はあまり差はない．

---

以下，実装における上記のアルゴリズムとの対応について述べる．

(1A) 平均が$\bx_{k - 1}^{(i)}$，(共)分散が$\bSigma_v$の正規分布でサンプリング

\begin{align}
    \tilde{\bx}\_k^{(i)} \sim \mathcal{N}(\bx_k|\bx_{k - 1}^{(i)}, \bSigma_v)
\end{align}

(2A) 平均が$\tilde{\bx}\_k^{(i)}$，(共)分散が$\bSigma_w$の正規分布で$\by_k$が観測される確率を尤度として算出．注意点は，桁落ち回避のために対数尤度を用いる点と，重みを算出するときは負の対数尤度の逆数を用いる点．

\begin{align}
    \tilde{w}_k^{(i)} \propto \mathcal{N} (\by_k|\tilde{\bx}_k^{(i)}, \bSigma_w)
\end{align}

(A3)および(A4)はアルゴリズムと同様である．

以下の映像は粒子の数$M = 50$，時刻$k = 50$までのアニメーション例である．粒子群による確率密度関数はカーネル密度推定で表現している．

<video controls>
    <source src="data/2018-11-11-monte-carlo-filter/monte_carlo_filter_example.mp4" type="video/mp4">
</video>

## 参考文献

* [21世紀の統計科学 Vol. III 数理・計算の統計科学](http://park.itc.u-tokyo.ac.jp/atstat/jss75shunen/Vol3.pdf)
