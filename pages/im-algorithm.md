---
title: "The IM Algorithm : A variational approach to Information Maximization (NIPS2003)"
tags: [GAN]
sidebar: home_sidebar
permalink: im-algorithm.html
---

\begin{align}
\newcommand{\b}[1]{\mathbf{#1}}
\newcommand{\bo}[1]{\boldsymbol{#1}} \nonumber
\end{align}

※式変形のメモです．

---

入力を$\mathbf{x}$，出力を$\mathbf{y}$とする．
相互情報量$I(\mathbf{x}, \mathbf{y})$，エントロピー$H(\mathbf{x})$，条件付きエントロピー$H(\mathbf{x} \| \mathbf{y})$について一般に以下が成り立つ．

$$
\newcommand{\b}[1]{\mathbf{#1}}
\newcommand{\bo}[1]{\boldsymbol{#1}}
I(\b{x}, \b{y}) = H(\b{x}) - H(\b{x}|\b{y}) = - \mathbb{E}_{\b{x} \sim p(\b{x})} \left[ \log p(\b{x})  \right] + \mathbb{E}_{\mathbb{x}, \mathbb{y} \sim p(\b{x}, \b{y})} \left[ \log p(\b{x} | \b{y}) \right]
$$

ここで，ある任意の確率分布$q(\mathbf{x} \| \mathbf{y}) \in Q$を導入すると，KLダイバージェンスの非負性から

$$
\newcommand{\b}[1]{\mathbf{#1}}
\newcommand{\bo}[1]{\boldsymbol{#1}}
\mathrm{KL} (p(\b{x}|\b{y}) \| q(\b{x}|\b{y})) = \mathbb{E}_{\b{x}, \b{y} \sim p(\b{x}, \b{y})} \left[ \log \frac{p(\b{x}|\b{y})}{q(\b{x}|\b{y})} \right] = \mathbb{E}_{\b{x}, \b{y} \sim p(\b{x}, \b{y})} \left[ \log p(\b{x}|\b{y}) \right] - \mathbb{E}_{\b{x}, \b{y} \sim p(\b{x}, \b{y})} \left[ \log q(\b{x}|\b{y}) \right] \geq 0 \\
\therefore \mathbb{E}_{\b{x}, \b{y} \sim p(\b{x}, \b{y})} \left[ \log p(\b{x}|\b{y}) \right] \geq \mathbb{E}_{\b{x}, \b{y} \sim p(\b{x}, \b{y})} \left[ \log q(\b{x}|\b{y}) \right]
$$

である．これを用いると相互情報量は

$$
\newcommand{\b}[1]{\mathbf{#1}}
\newcommand{\bo}[1]{\boldsymbol{#1}}
I(\b{x}, \b{y}) = H(\b{x}) - \mathbb{E}_{\mathbb{x}, \mathbb{y} \sim p(\b{x}, \b{y})} \left[ \log p(\b{x} | \b{y}) \right] \geq H(\b{x}) + \mathbb{E}_{\b{x}, \b{y} \sim p(\b{x}, \b{y})} \left[ \log q(\b{x}|\b{y}) \right]
\overset{\mathrm{def}}{=} \tilde{I}(\b{x}, \b{y})
$$

というように下からバウンドできる．

## IM algorithm

1. $q(\mathbf{x} \| \mathbf{y})$を固定．$\theta$を$\theta^{new} = \argmax_{\theta} \tilde{I}(\mathbf{x}, \mathbf{y})$で更新．
2. $\theta$を固定．$q(\mathbf{x} \| \mathbf{y})$を$q^{new}(\mathbf{x} \| \mathbf{y}) = \argmax_{q(\mathbf{x} \| \mathbf{y}) \in Q} \tilde{I}(\mathbf{x}, \mathbf{y})$で更新．

## Relation to Conditional Likelihood

$x \rightarrow y \rightarrow \tilde{x}$というautoencoderを考える．autoencoderは入力$x$が値$s$をとるなら出力$\tilde{x}$も同様に値$s$をとる確率を最大化したい．つまり

$$
\log p(\tilde{x} = s | x = s) = \log \left( \int p(\tilde{x} = s, y | x = s) dy \right) = \\
\log \left( \int p(\tilde{x} = s | y, x = s) p(y|x = s) dy \right) = \log \left( \int p(\tilde{x} = s | y) p(y|x = s) dy \right) \\
\overbrace{\geq}^{Jensen} \int \log p(\tilde{x} = s | y) p(y|x = s) dy = \mathbb{E}_{y \sim p(y|x = s)} \left[ \log p(\tilde{x} = s | y) \right]
$$

途中の$p(\tilde{x} = s \| y, x = s) = p(\tilde{x} = s \| y)$は$x \rightarrow y \rightarrow \tilde{x}$という有向グラフィカルモデルを考えることで得られる．$y$で条件付けると，$x$から$\tilde{x}$への経路でhead-to-tailにあたる$y$が観測されたことになるので条件付き独立性が成立するからである．

$s$についての期待値をとると

$$
\sum_s p(x = s) \log p(\tilde{x} = s | x = s) \geq \sum_s p(x = s) \mathbb{E}_{y \sim p(y|x = s)} \left[ \log p(\tilde{x} = s | y) \right] \\ 
= \sum_s \mathbb{E}_{y \sim p(x = s, y)} \left[ \log p(\tilde{x} = s | y) \right] \equiv \mathbb{E}_{x, y \sim p(x, y)} \left[ \log p(\tilde{x} | y) \right]
$$

を得る．即ち，$\tilde{I}(\b{x}, \b{y})$は$p(\b{x})$を固定すると，再構成についてのlower boundになっている．

## References

- [The IM Algorithm : A variational approach to Information Maximization](https://papers.nips.cc/paper/2410-information-maximization-in-noisy-channels-a-variational-approach.pdf)