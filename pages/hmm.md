---
title: HMMのEMアルゴリズムによる最尤推定
tags: [HMM]
sidebar: home_sidebar
permalink: hmm.html
---

\begin{align}
\newcommand{\cS}{\mathcal{S}}
\newcommand{\bt}{\mathbf{t}}
\newcommand{\bx}{\mathbf{x}}
\newcommand{\by}{\mathbf{y}}
\newcommand{\bw}{\mathbf{w}}
\newcommand{\bQ}{\mathbf{Q}}
\newcommand{\b}[1]{\mathbf{#1}}
\newcommand{\bo}[1]{\boldsymbol{#1}}
\newcommand{\old}{\mathrm{old}}
\newcommand{\new}{\mathrm{new}}
\newcommand{\bphi}{\boldsymbol{\phi}}
\newcommand{\balpha}{\boldsymbol{\alpha}}
\newcommand{\bbeta}{\boldsymbol{\beta}}
\newcommand{\bgamma}{\boldsymbol{\gamma}}
\newcommand{\bxi}{\boldsymbol{\xi}}
\newcommand{\bSigma}{\boldsymbol{\Sigma}} \nonumber
\end{align}

<!--

TODO: グラフィカルモデルの図を追加．

HMMのEMアルゴリズムによる最尤推定について述べる．

**GOAL: PRML中で省略されている式変形を補う．**

## 前準備

一次マルコフ性を持つと仮定して，進めていく．

### 記号表記と制約

- 観測変数：$\b{X} = \left\\{ \b{x}\_1, \ldots, \b{x}\_N \right\\}$
- 潜在変数：$\b{Z} = \left\\{ \b{z}\_1, \ldots, \b{z}\_N \right\\}$
- $p(\b{z}\_1)$のパラメータ：$\bo{\pi} = \left[ \pi_1, \ldots, \pi_K \right]^T \quad \left( \sum_k \pi_k = 1 \right)$
- 各成分が遷移確率を表す行列：$\b{A} = \left(A_{jk} \right)_{1 \leq j \leq K, 1 \leq k \leq K} \quad \left( \sum_k A\_{jk} = 1 \right)$
- $p(\b{x} \| \b{z})$のパラメータ：$\bo{\phi} = \left\\{ \bo{\phi}\_1, \ldots, \bo{\phi}_K \right\\}$
- パラメータ集合：$\bo{\theta} = \left\\{ \bo{\pi}, \b{A}, \bo{\phi} \right\\}$

潜在変数は離散変数だとしている．遷移確率は
\begin{align}
A_{jk} = p(z_{nk} = 1) | z_{n - 1, j} = 1)
\end{align}
と定義される．

## EMアルゴリズムによる最尤推定

まず$Q$関数は以下のように定義される．

\begin{align}
Q(\bo{\theta}, \bo{\theta}^{\old}) = \sum_{\b{Z}} p(\b{Z} | \b{X}, \bo{\theta}^{\old}) \ln p(\b{X}, \b{Z} | \bo{\theta})
\end{align}

Eステップでは，$p(\b{Z} | \b{X}, \bo{\theta}^{\old})$を求める．
Mステップでは，$Q(\bo{\theta}, \bo{\theta}^{\old})$を最大化する$\bo{\theta}$を$\bo{\theta}^{\new}$とする．

---

HMMでは潜在変数を含めた同時確率はグラフィカルモデルから以下のように分解できることが分かる．

\begin{align}
p(\b{X}, \b{Z} | \bo{\theta}) = p(\b{z}\_1 | \bo{\pi}) \left[ \prod_{n = 2}^N p(\b{z}\_n | \b{z}\_{n - 1}, \b{A}) \right] \left[ \prod_{m = 1}^N p(\b{x}\_m | \b{z}\_m, \bo{\phi}) \right]
\end{align}

ここで，各確率分布はパラメータを用いて表記すると以下のようになる．

\begin{align}
p(\b{z}\_{1} | \bo{\pi}) &= \prod_{k = 1}^K \pi_k^{z_{1k}} \newline
p(\b{z}\_n | \b{z}\_{n - 1}, \b{A}) &= \prod_{k = 1}^K \prod_{j = 1}^K A_{jk}^{z_{n - 1, j} \; z_{nk}} \newline
p(\b{x}\_n | \b{z}\_n, \bo{\phi}) &= \prod_{k = 1}^K p(\b{x}\_n | \bo{\phi}_k)^{z\_{nk}}
\end{align}

これらを用いて，EステップおよびMステップを示そう．

### Eステップ

\begin{align}
& Q(\bo{\theta}, \bo{\theta}^{\old}) \newline
&= \sum_{\b{Z}} p(\b{Z} | \b{X}, \bo{\theta}^{\old}) \ln p(\b{X}, \b{Z} | \bo{\theta}) \newline
&= \sum_{\b{Z}} p(\b{Z} | \b{X}, \bo{\theta}^{\old}) \left\\{ \ln p(\b{z}\_1 | \bo{\pi} ) + \sum\_{n = 2}^N \ln p(\b{z}\_n | \b{z}\_{n - 1}, \b{A}) + \sum\_{n = 1}^N \ln p(\b{x}\_n | \bo{\phi}\_k) \right\\} \newline
&= \sum_{\b{Z}} p(\b{Z} | \b{X}, \bo{\theta}^{\old}) \left\\{ \sum_{k = 1}^K z_{1k} \ln \pi_k + \sum_{n = 2}^N \sum_{j = 1}^K \sum_{k = 1}^K z_{n - 1, j} z_{nk} \ln A_{jk} + \sum_{n = 1}^N \sum_{k = 1}^K z_{nk} \ln p(\b{x}\_n | \bo{\phi}\_k) \right\\} \newline
&= \sum\_{k = 1}^K \mathbb{E} \left[ z_{1k} \right] \ln \pi_k + \sum_{n = 2}^N \sum\_{j = 1}^K \sum\_{k = 1}^K \mathbb{E} \left[ z_{n - 1, j} z_{nk} \right] \ln A_{jk} + \sum_{n = 1}^N \sum\_{k = 1}^K \mathbb{E} \left[ z_{nk} \right] \ln p(\b{x}\_n | \bo{\phi}_k)
\end{align}

ここで，PRMLと同様にいくつかの記号を導入する．$\b{z}\_n$の周辺事後分布を$\gamma(\b{z}_n) = p(\b{z}_n \| \b{X}, \bo{\theta}^{\old})$とし，$\b{z}\_{n - 1}, \b{z}\_n$の周辺事後分布を$\xi(\b{z}\_{n - 1}, \b{z}\_n) = p(\b{z}\_{n - 1}, \b{z}_n \| \b{X}, \bo{\theta}^{\old})$とする．さらに，$z\_{nk} = 1$の事後確率を$\gamma(z\_{nk})$とし，$z\_{n - 1, j} = 1, z\_{nk} = 1$の事後確率を$\xi(z\_{n - 1, j}, z\_{nk})$とすると，以下のように期待値と一致する．

\begin{align}
\gamma(z\_{nk}) &= \sum_{\b{z}\_n} \gamma(\b{z}\_n) z_{nk} = \mathbb{E} \left[ z_{nk} \right] \newline
\xi(z\_{n - 1, j}, z\_{nk}) &= \sum_{\b{z}\_{n - 1}, \b{z}\_n} \xi(\b{z}\_{n - 1}, \b{z}\_n) z\_{n - 1, j} z\_{nk} = \mathbb{E} \left[ z\_{n - 1, j}, z\_{nk} \right]
\end{align}

これらの記号を用いると，$Q$関数は

\begin{align}
Q(\bo{\theta}, \bo{\theta}^{\old}) 
= \sum\_{k = 1}^K \gamma (z_{1k}) \ln \pi_k + \sum\_{k = 1}^K \sum\_{j = 1}^K \xi (z_{n - 1, j}, z_{nk}) \ln A_{jk} + \sum\_{k = 1}^K \gamma (z_{nk}) \ln p(\b{x}\_n | \bo{\phi}_k)
\end{align}

となる．

$\gamma (z_{1k}), \gamma (z_{nk}), \xi (z_{n - 1, j}, z_{nk})$を求める必要があるが，これは後述するforward-backwardアルゴリズムあるいはBaum-Welchアルゴリズムによって求める．

### Mステップ

Mステップでは，$\gamma(\b{z}\_n)$と$\xi(\b{z}\_{n - 1}, \b{z}\_n)$を定数とみなして，パラメータ$\bo{\theta}$について$Q(\bo{\theta}, \bo{\theta}^{\old})$を最大化する．これは，以下の制約付き最適化問題とみなすことができる．

**TODO: ここに最適化問題を書く**

まず，$\bo{\pi}$と$\b{A}$について最大化する．上式においてラグランジュの未定乗数法を用いると，以下の無制約最適化問題が得られる．

\begin{align}
L &= Q(\bo{\theta}, \bo{\theta}^{\old}) - \alpha_1 \left(\sum_{k = 1}^K A_{1k} - 1 \right) - \cdots - \alpha_K \left(\sum_{k = 1}^K A_{Kk} - 1 \right) - \beta \left(\sum_{k = 1}^K \pi_k - 1 \right) \newline
&= Q(\bo{\theta}, \bo{\theta}^{\old}) - \sum_{j = 1}^K \alpha_j \left(\sum_{k = 1}^K A_{jk} - 1 \right) - \beta \left(\sum_{k = 1}^K \pi_k - 1 \right)
\end{align}
ただし，$\alpha_1, \ldots, \alpha_K, \beta$はラグランジュ乗数である．ここから$A_{jk}, \pi_k$の偏微分を$0$とおいてラグランジュ乗数を求めよう．

\begin{align}
\frac{\partial L}{\partial A_{jk}} = \sum_{n = 2}^N \frac{1}{A_{jk}} \xi(z\_{n - 1, j}, z\_{nk}) - \alpha_j = 0 \newline
\alpha_j A_{jk} = \sum_{n = 2}^N \xi(z\_{n - 1, j}, z\_{nk})
\end{align}
両辺を$k$について総和をとると，制約$\sum_{k = 1}^K A_{jk} = 1$より
\begin{align}
\sum_{k = 1}^K \alpha_j A_{jk} = \alpha_j = \sum_{k = 1}^K \sum_{n = 2}^N \xi(z\_{n - 1, j}, z\_{nk})
\end{align}

を得る．この$\alpha_j$を$\frac{\partial L}{\partial A_{jk}} = 0$に代入することで，以下を得る．ただし，このとき$j, k$はある特定の添字を選択しているので，$\alpha_j$の等式の右辺で総和をとっている添字$k$を$l$に変更することで添字の混乱を防ぐ．つまり，$\alpha_j = \sum_{l = 1}^K \sum_{n = 2}^N \xi(z\_{n - 1, j}, z\_{nl})$として

\begin{align}
\sum_{n = 2}^N \frac{1}{A_{jk}} \xi(z\_{n - 1, j}, z\_{nk}) = \alpha_j \newline
\sum_{n = 2}^N \xi(z\_{n - 1, j}, z\_{nk}) = A_{jk} \sum_{l = 1}^K \sum_{n = 2}^N \xi(z\_{n - 1, j}, z\_{nl}) \newline
A_{jk} = \frac{\sum_{n = 2}^N \xi(z\_{n - 1, j}, z\_{nk})}{\sum_{l = 1}^K \sum_{n = 2}^N \xi(z\_{n - 1, j}, z\_{nl})}
\end{align}

となる．次に$\pi_k$について求める．

\begin{align}
\frac{\partial L}{\partial \pi_k} = \frac{1}{\pi_k} \gamma (z_{1k}) - \beta = 0 \newline
\beta \pi_k = \gamma (z_{1k})
\end{align}

両辺を$k$について総和をとると，制約$\sum_{k = 1}^K \pi_k = 1$より

\begin{align}
\sum_{k = 1}^K \beta \pi_k = \beta = \sum_{k = 1}^K \gamma (z_{1k})
\end{align}

である．この$\beta$を$\frac{\partial L}{\partial \pi_k} = 0$に代入する．$A_{jk}$のときと同様に，$\beta$の等式の右辺の添字$k$を$l$に変更しておく．つまり$\beta = \sum_{l = l}^K \gamma (z_{1l})$として

\begin{align}
\frac{1}{\pi_k} \gamma (z_{1k}) = \beta \newline
\gamma (z_{1k}) = \pi_k \sum_{l = l}^K \gamma (z_{1l}) \newline
\pi_k = \frac{\gamma (z_{1k})}{\sum_{l = l}^K \gamma (z_{1l})}
\end{align}

となる．

最後に$\bo{\phi}$について最大化する．PRMLと同様に，$p(\b{x} \| \bo{\phi}_k) = \mathcal{N}(\b{x} \| \bo{\mu}_k, \bo{\Sigma}_k)$の場合を考える．これはGMMにEMアルゴリズムを適用するときに部分的に解いている．よって

\begin{align}
\bo{\mu}\_k &= \frac{\sum\_{n = 1}^N \gamma (z_{nk}) \b{x}\_n}{\sum\_{n = 1}^N \gamma (z_{nk})} \newline
\bo{\Sigma}\_k &= \frac{\sum\_{n = 1}^N \gamma (z_{nk}) (\b{x}\_n - \bo{\mu}\_k) (\b{x}\_n - \bo{\mu}\_k)^T }{\sum\_{n = 1}^N \gamma (z_{nk})}
\end{align}
である．

-->

## Baum-Welchアルゴリズム
### 条件付き独立性

Baum-Welchアルゴリズムの導出過程で必要となる条件付き独立性の式を示す．ここではPRMLの1つ目の条件付き独立性のみ示しておく．

![fig1](data/hmm/fig1.png)

$\b{x}\_{1:n}$から$\b{x}\_{n + 1:N}$への経路は$\b{z}\_{n}$によって遮断されている．よって条件付き独立性により$p(\b{x}\_{1:N} \| \b{z}\_n) = p(\b{x}\_{1:n} \| \b{z}\_n) p(\b{x}\_{n + 1:N} \| \b{z}\_n)$が成り立つ．

他の条件付き独立性は以下の通り．
- $p(\b{x}\_{1:n - 1} \| \b{x}\_n, \b{z}\_n) = p(\b{x}\_{1:n - 1} \| \b{z}\_n)$
- $p(\b{x}\_{1:n - 1} \| \b{z}\_{n - 1}, \b{z}\_n) = p(\b{x}\_{1:n - 1} \| \b{z}\_{n - 1})$
- $p(\b{x}\_{n + 1:N} \| \b{z}\_n, \b{z}\_{n + 1}) = p(\b{x}\_{n + 1:N} \| \b{z}\_{n + 1})$
- $p(\b{x}\_{n + 2:N} \| \b{z}\_{n + 1}, \b{x}\_{n + 1}) = p(\b{x}\_{n + 2:N} \| \b{z}\_{n + 1})$
- $p(\b{x}\_{1:N} \| \b{z}\_{n - 1}, \b{z}\_n) = p(\b{x}\_{1:n - 1} \| \b{z}\_{n - 1}) p(\b{x}\_n \| \b{z}\_n) p(\b{x}\_{n + 1:N} \| \b{z}\_n)$
- $p(\b{x}_{N + 1} \| \b{x}\_{1:N}, \b{z}\_{N + 1}) = p(\b{x}\_{N + 1} \| \b{z}\_{N + 1})$
- $p(\b{z}_{N + 1} \| \b{z}\_N, \b{x}\_{1:N}) = p(\b{z}\_{N + 1} \| \b{z}\_N)$

---

<!-- \newcommand{\b}[1]{\mathbf{#1}}
\newcommand{\bo}[1]{\boldsymbol{#1}} -->

$\alpha(\b{z}\_n) := p(\b{x}\_{1:n}, \b{z}\_n), \beta(\b{z}\_n) := p(\b{x}\_{n + 1:N} \| \b{z}\_n)$と定義すると，

\begin{align}
\gamma(\b{z}\_n) 
&= p(\b{z}\_n | \b{x}\_{1:N}) = \frac{p(\b{z}\_n, \b{x}\_{1:N})}{p(\b{x}\_{1:N})}
= \frac{p(\b{x}\_{1:N} | \b{z}\_n) p(\b{z}\_n)}{p(\b{x}\_{1:N})}
= \frac{p(\b{x}\_{1:n} | \b{z}\_n) p(\b{x}\_{n + 1:N} | \b{z}\_n) p(\b{z}\_n)}{p(\b{x}\_{1:N})} \newline
&= \frac{\overbrace{p(\b{x}\_{1:n}, \b{z}\_n)}^{= \alpha(\b{z}\_n)} \overbrace{ p(\b{x}\_{n + 1:N} | \b{z}\_n)}^{= \beta(\b{z}\_n)} }{p(\b{x}\_{1:N})} = \frac{\alpha(\b{z}\_n) \beta(\b{z}\_n)}{p(\b{x}\_{1:N})}
\end{align}

である．さらに$\alpha, \beta$について再帰式を導出する．

\begin{align}
\alpha(\b{z}\_n) 
&= p(\b{x}\_{1:n},  \b{z}\_n) = p(\b{x}\_{1:n} | \b{z}\_n) p(\b{z_n})
= p(\b{x}\_n | \b{z}\_n) \underbrace{p(\b{x}\_{1:n - 1} | \b{x}\_n, \b{z}\_n)}_{= p(\b{x}\_{1:n - 1} | \b{z}\_n)} p(\b{z}_n)
= p(\b{x}\_n | \b{z}\_n) p(\b{x}\_{1:n - 1} | \b{z}\_n) p(\b{z}\_n) \newline
&= p(\b{x}\_n | \b{z}\_n) p(\b{x}\_{1:n - 1}, \b{z}\_n)
= p(\b{x}\_n | \b{z}\_n) \sum\_{\b{z}\_{n - 1}} p(\b{x}\_{1:n - 1}, \b{z}\_n, \b{z}\_{n - 1})
= p(\b{x}\_n | \b{z}\_n) \sum\_{\b{z}\_{n - 1}} p(\b{x}\_{1:n - 1}, \b{z}\_n, \b{z}\_{n - 1}) \newline
&= p(\b{x}\_n | \b{z}\_n) \sum\_{\b{z}\_{n - 1}} p(\b{x}\_{1:n - 1}, \b{z}\_n | \b{z}\_{n - 1}) p(\b{z}\_{n - 1})
= p(\b{x}\_n | \b{z}\_n) \sum\_{\b{z}\_{n - 1}} \underbrace{p(\b{x}\_{1:n - 1}, | \b{z}\_n, \b{z}\_{n - 1})}\_{= p(\b{x}\_{1:n - 1}, | \b{z}\_{n - 1})} p(\b{z}\_n | \b{z}\_{n - 1}) p(\b{z}\_{n - 1}) \newline
&= p(\b{x}\_n | \b{z}\_n) \sum\_{\b{z}\_{n - 1}} p(\b{x}\_{1:n - 1}, | \b{z}\_{n - 1}) p(\b{z}\_n | \b{z}\_{n - 1}) p(\b{z}\_{n - 1}) \newline
&= p(\b{x}\_n | \b{z}\_n) \sum\_{\b{z}\_{n - 1}} \underbrace{p(\b{x}\_{1:n - 1}, \b{z}\_{n - 1})}\_{= \alpha(\b{z}\_{n - 1})} p(\b{z}\_n | \b{z}\_{n - 1})
= p(\b{x}\_n | \b{z}\_n) \sum\_{\b{z}\_{n - 1}} \alpha(\b{z}\_{n - 1}) p(\b{z}\_n | \b{z}\_{n - 1})
\end{align}

したがって，$\alpha$について再帰的な式を得る．
\begin{align}
\alpha(\b{z}\_n) = p(\b{x}\_n | \b{z}\_n) \sum\_{\b{z}\_{n - 1}} \alpha(\b{z}\_{n - 1}) p(\b{z}\_n | \b{z}\_{n - 1})
\end{align}

---
