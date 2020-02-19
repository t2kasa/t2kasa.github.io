---
title: "LDAで崩壊型ギブスサンプリング"
permalink: 2020-02-21-lda-collapsed-gibbs.html
sidebar: blog_sidebar
---

\begin{align}
\newcommand{\b}[1]{\mathbf{#1}}
\newcommand{\bo}[1]{\boldsymbol{#1}} \nonumber
\newcommand{\bs}{\backslash}
\end{align}

**※今回の記事は実装を含んでいません．**

[ベイズ推論による機械学習入門][bayes-book]より，LDAによる崩壊型ギブスサンプリングを見ていきます．これは5.4節の内容です．事前に前回の記事を参照してください．記号の表記やモデル等は同一です．

さて，今回はLDAで崩壊型ギブスサンプリングです．パラメータ$\bo{\Theta}, \bo{\Phi}$を周辺化除去し，$\b{Z}$についてのギブスサンプリングのアルゴリズムを導出することが目標です．

まず，$\bo{\Theta}$で周辺化したときの変数間の依存関係について確認してみます．そのため，$\bo{\theta}_d$と依存関係がある任意の異なる添字$i, j \; (i \neq j)$を持つ2つのノード$\b{z}\_{d, i}, \b{z}\_{d, j}$の箇所のみを取り出したグラフィカルモデルを表してみます．すると，ノード間の関係性はtail-to-tailとなっていることから，$\bo{\theta}_d$で周辺化すると$\b{z}\_{d, i}, \b{z}\_{d, j}$間に依存関係を持つようになることが分かります．一方，$d$と異なる$d'$のグラフィカルモデルも同時に表現すると，添字に$d, d'$を持つノード間では接続がないことから，$\bo{\Theta}$で周辺化してもこれらのノード間は独立であることが分かります．

![](data/2020-02-21-lda-collapsed-gibbs/fig1.png){:style="width: auto; height: 400px"}

次は$\bo{\Phi}$で周辺化したときの変数間の依存関係を確認します．
$\bo{\phi}\_k, \bo{w}\_{d, i}, \bo{w}\_{d, j}$に着目すると，tail-to-tailなので先程と同様に，$\bo{\phi}\_k$で周辺化するとノード間に依存関係を持つようになります．更に，$\bo{\phi}\_k, \bo{w}\_{d, i}, \bo{w}\_{d', i}$でもtail-to-tailとなっています．したがって，$\bo{w}_{d, n}$は任意の異なる添字のノード間で依存関係を持つことが分かります．$\bo{\phi}\_{k'}$についても同様です．

![](data/2020-02-21-lda-collapsed-gibbs/fig2.png){:style="width: auto; height: 400px"}

これまでの結果を踏まえると，$\bo{\Theta}, \bo{\Phi}$で周辺化したときのグラフィカルモデルは以下のようになります．ただし，青色の丸みを帯びた枠で囲んだ内部のノード間は全て依存関係を持つ完全グラフになっています（表記の煩雑さを軽減するため，枠を用いています）．

![](data/2020-02-21-lda-collapsed-gibbs/lda-marginalization.png)

## 崩壊型ギブスサンプリング

まず，新たな記号表記を追加します．

- 文書$d$における潜在変数の集合$\b{Z}_d$から$\b{z}\_{d, n}$のみを除いた部分集合$\b{Z}\_{d, \bs n}$
- 潜在変数の全体$\b{Z}$から$\b{Z}\_d$を除いた部分集合$\b{Z}_{\bs d}$
- 文書集合$\b{W}$に対する上記2つと同様の表記として$\b{W}\_{d, \bs n}, \b{W}_{\bs d}$

この表記を用いると，$\b{Z} = \left\\{ \b{z}\_{d, n}, \b{Z}\_{d, \bs n}, \b{Z}\_{\bs d} \right\\}, \b{W} = \left\\{ \b{w}\_{d, n}, \b{W}\_{d, \bs n}, \b{W}\_{\bs d} \right\\}$となります．

さて，崩壊型ギブスサンプリングです．周辺化したモデルで$\b{z}\_{d, n}$の条件付き分布を考えていきます．

\begin{align}
& p(\b{z}\_{d, n} | \b{Z}\_{d, n}, \b{Z}\_{\bs d}, \b{W}) \newline
&\propto p(\b{w}\_{d, n}, \b{W}\_{d, \bs n}, \b{W}\_{\bs d}, \b{z}\_{d, n}, \b{Z}\_{d, \bs n}, \b{Z}\_{\bs d}) \newline
&= p(\b{w}\_{d, n} | \b{W}\_{d, \bs n}, \b{W}\_{\bs d}, \b{z}\_{d, n}, \b{Z}\_{d, \bs n}, \b{Z}\_{\bs d}) p(\b{W}\_{d, \bs n} | \b{W}\_{\bs d}, \b{z}\_{d, n}, \b{Z}\_{d, \bs n}, \b{Z}\_{\bs d}) p(\b{W}\_{\bs d} | \b{z}\_{d, n}, \b{Z}\_{d, \bs n}, \b{Z}\_{\bs d}) p(\b{z}\_{d, n} | \b{Z}\_{d, \bs n}, \b{Z}\_{\bs d}) p(\b{Z}\_{d, \bs n}, \b{Z}\_{\bs d})
\end{align}

この分解は単純に条件付き分布の定義に従って行ったものです．ここで，分解した各条件付き分布について変数間の独立性をグラフィカルモデルを用いて見ていきます．

---

この分布はこれ以上簡単になりません．

\begin{align}
p(\b{w}\_{d, n} | \b{W}\_{d, \bs n}, \b{W}\_{\bs d}, \b{z}\_{d, n}, \b{Z}\_{d, \bs n}, \b{Z}\_{\bs d})
\end{align}

![](data/2020-02-21-lda-collapsed-gibbs/cond1-1.png)

---

この分布は条件から$\b{z}\_{d, n}$を削除できます．

\begin{align}
p(\b{W}\_{d, \bs n} | \b{W}\_{\bs d}, \b{z}\_{d, n}, \b{Z}\_{d, \bs n}, \b{Z}\_{\bs d}) = p(\b{W}\_{d, \bs n} | \b{W}\_{\bs d}, \b{Z}\_{d, \bs n}, \b{Z}\_{\bs d})
\end{align}

![](data/2020-02-21-lda-collapsed-gibbs/cond1-2.png)

---

この分布は条件から$\b{z}\_{d, n}, \b{Z}\_{d, \bs n}$を削除できます．

\begin{align}
p(\b{W}\_{\bs d} | \b{z}\_{d, n}, \b{Z}\_{d, \bs n}, \b{Z}\_{\bs d}) = p(\b{W}\_{\bs d} | \b{Z}\_{\bs d})
\end{align}

![](data/2020-02-21-lda-collapsed-gibbs/cond1-3.png)

---

この分布は条件から$\b{Z}\_{\bs d}$を削除できます．

\begin{align}
p(\b{z}\_{d, n} | \b{Z}\_{d, \bs n}, \b{Z}\_{\bs d}) = p(\b{z}\_{d, n} | \b{Z}\_{d, \bs n})
\end{align}

![](data/2020-02-21-lda-collapsed-gibbs/cond1-4.png)

---

この分布はそもそも条件がないのでそのままです．

\begin{align}
p(\b{Z}\_{d, \bs n}, \b{Z}\_{\bs d})
\end{align}

![](data/2020-02-21-lda-collapsed-gibbs/cond1-5.png)

---

以上を踏まえて条件付き分布を簡単にした後，$\b{z}\_{d, n}$を含まない分布を定数とみなします．

\begin{align}
& p(\b{z}\_{d, n} | \b{Z}\_{d, n}, \b{Z}\_{\bs d}, \b{W}) \newline
&\propto p(\b{w}\_{d, n} | \b{W}\_{d, \bs n}, \b{W}\_{\bs d}, \b{z}\_{d, n}, \b{Z}\_{d, \bs n}, \b{Z}\_{\bs d}) p(\b{W}\_{d, \bs n} | \b{W}\_{\bs d}, \b{Z}\_{d, \bs n}, \b{Z}\_{\bs d}) p(\b{W}\_{\bs d} | \b{Z}\_{\bs d}) p(\b{z}\_{d, n} | \b{Z}\_{d, \bs n}) p(\b{Z}\_{d, \bs n}, \b{Z}\_{\bs d}) \newline
&\propto p(\b{w}\_{d, n} | \b{W}\_{d, \bs n}, \b{W}\_{\bs d}, \b{z}\_{d, n}, \b{Z}\_{d, \bs n}, \b{Z}\_{\bs d}) p(\b{z}\_{d, n} | \b{Z}\_{d, \bs n})
\end{align}

あとは残った2つの分布について，考えていきます．ここで，$k = 1, \ldots, K$のそれぞれに対して$z\_{d, n, k} = 1$のときの値を計算できれば，それを正規化することで$\b{z}\_{d, n}$についてのカテゴリ分布の確率とすることができます．この点を踏まえて，2つの分布について計算していきます．

$p(\b{z}\_{d, n} \| \b{Z}\_{d, \bs n})$は$\b{z}\_{d, n}$の予測分布とみなすことで

\begin{align}
p(\b{z}\_{d, n} | \b{Z}\_{d, \bs n}) = \int p(\b{z}\_{d, n} | \bo{\theta}_d) p(\bo{\theta}_d | \bo{Z}\_{d, \bs n}) d \bo{\theta}_d
\end{align}

と表現できます．ここで$p(\bo{\theta}_d \| \bo{Z}\_{d, \bs n})$について計算してきます．

\begin{align}
p(\bo{\theta}_d | \b{Z}\_{d, \bs n}) 
&\propto p(\bo{\theta}_d, \b{Z}\_{d, \bs n}) \newline
&= p(\b{Z}\_{d, \bs n} | \bo{\theta}_d) p(\bo{\theta}_d) \newline
&= \left\\{ \prod\_{n' \neq n} p(\b{z}\_{d, n'} | \bo{\theta}_d) \right\\} p(\bo{\theta}_d) \newline
&= \left\\{ \prod\_{n' \neq n} \mathrm{Cat} (\b{z}\_{d, n'} | \bo{\theta}_d) \right\\} \mathrm{Dir} (\bo{\theta}_d | \bo{\alpha})
\end{align}

対数をとると

\begin{align}
\ln p(\bo{\theta}_d | \b{Z}\_{d, \bs n}) 
&= \sum\_{n' \neq n} \ln \mathrm{Cat} (\b{z}\_{d, n'} | \bo{\theta}_d) + \ln \mathrm{Dir} (\bo{\theta}_d | \bo{\alpha}) \newline
&= \sum\_{n' \neq n} \sum\_{k = 1}^K z\_{d, n', k} \ln \theta\_{d, k} + \sum\_{k = 1}^K (\alpha_k - 1) \ln \theta\_{d, k} + \mathrm{const.} \newline
&= \sum\_{k = 1}^K \left( \sum\_{n' \neq n} z\_{d, n', k} + \alpha_k - 1 \right) \ln \theta\_{d, k} + \mathrm{const.}
\end{align}

となるので，これはディリクレ分布となることが分かります．

\begin{align}
p(\bo{\theta}_d | \b{Z}\_{d, \bs n}) = \mathrm{Dir} (\bo{\theta}_d | \tilde{\bo{\alpha}}\_{d, \bs n}) \quad \left(\tilde{\alpha}\_{d, \bs n, k} = \sum\_{n' \neq n} z\_{d, n', k} + \alpha_k \right)
\end{align}

したがって，$p(\b{z}\_{d, n} \| \b{Z}\_{d, \bs n})$は

\begin{align}
p(\b{z}\_{d, n} | \b{Z}\_{d, \bs n}) = \int \mathrm{Cat} (\b{z}\_{d, n} | \bo{\theta}_d) \mathrm{Dir} (\bo{\theta}_d | \tilde{\bo{\alpha}}\_{d, \bs n} ) d \bo{\theta}_d = \mathrm{Cat} (\b{z}\_{d, \bs n} | \hat{\bo{\alpha}}\_{d, \bs n})
\end{align}

\begin{align}
\hat{\alpha}\_{d, \bs n, k} \propto \sum\_{n' \neq n} z\_{d, n', k} + \alpha_k \quad \left( \mathrm{s.t.} \sum\_{k = 1}^K \hat{\alpha}\_{d, \bs n, k} = 1 \right)
\end{align}

となります．

次は$p(\b{w}\_{d, n} \| \b{W}\_{d, \bs n}, \b{W}\_{\bs d}, \b{z}\_{d, n}, \b{Z}\_{d, \bs n}, \b{Z}\_{\bs d})$についてです．
$\b{w}\_{d, n}$の予測分布とみなすことで

\begin{align}
p(\b{w}\_{d, n} | \b{W}\_{d, \bs n}, \b{W}\_{\bs d}, \b{z}\_{d, n}, \b{Z}\_{d, \bs n}, \b{Z}\_{\bs d}) = \int p(\b{w}\_{d, n} | \b{z}\_{d, n}, \bo{\Phi}) p(\bo{\Phi} | \b{W}\_{d, \bs n}, \b{W}\_{\bs d}, \b{Z}\_{d, \bs n}, \b{Z}\_{\bs d}) d \bo{\Phi}
\end{align}

とできます．

---

上記の式変形を補足します．まず$\bo{\Phi}$で周辺化しているので

\begin{align}
p(\b{w}\_{d, n} | \b{W}\_{d, \bs n}, \b{W}\_{\bs d}, \b{z}\_{d, n}, \b{Z}\_{d, \bs n}, \b{Z}\_{\bs d}) = \int p(\b{w}\_{d, n}, \bo{\Phi} | \b{W}\_{d, \bs n}, \b{W}\_{\bs d}, \b{z}\_{d, n}, \b{Z}\_{d, \bs n}, \b{Z}\_{\bs d}) d \bo{\Phi}
\end{align}

となっています．ここで，周辺化の対象となる分布は条件付き分布の定義から

\begin{align}
& p(\b{w}\_{d, n}, \bo{\Phi} | \b{W}\_{d, \bs n}, \b{W}\_{\bs d}, \b{z}\_{d, n}, \b{Z}\_{d, \bs n}, \b{Z}\_{\bs d}) \newline
&= p(\b{w}\_{d, n} | \bo{\Phi}, \b{W}\_{d, \bs n}, \b{W}\_{\bs d}, \b{z}\_{d, n}, \b{Z}\_{d, \bs n}, \b{Z}\_{\bs d}) p(\bo{\Phi} | \b{W}\_{d, \bs n}, \b{W}\_{\bs d}, \b{z}\_{d, n}, \b{Z}\_{d, \bs n}, \b{Z}\_{\bs d})
\end{align}

とできます．前半の$p(\b{w}\_{d, n} \| \bo{\Phi}, \b{W}\_{d, \bs n}, \b{W}\_{\bs d}, \b{z}\_{d, n}, \b{Z}\_{d, \bs n}, \b{Z}\_{\bs d})$
をグラフィカルモデルで表現してみます．ここでは$\bo{\Phi}$で周辺化していないので，異なる$\bo{w}_{d, n}$間で依存関係をもっていません．

![](data/2020-02-21-lda-collapsed-gibbs/cond2-1.png)

ここから，依存関係を持つのは$\b{z}\_{d, n}, \bo{\Phi}$だけであることが分かります．

\begin{align}
p(\b{w}\_{d, n} | \bo{\Phi}, \b{W}\_{d, \bs n}, \b{W}\_{\bs d}, \b{z}\_{d, n}, \b{Z}\_{d, \bs n}, \b{Z}\_{\bs d}) = p(\b{w}\_{d, n}  | \b{z}\_{d, n}, \bo{\Phi})
\end{align}

後半の$p(\bo{\Phi} \| \b{W}\_{d, \bs n}, \b{W}\_{\bs d}, \b{z}\_{d, n}, \b{Z}\_{d, \bs n}, \b{Z}\_{\bs d})$についても，同様にグラフィカルモデルで表現してみます．ここから，$\bo{\Phi}$と依存関係を持つのは$\b{w}\_{d', n'} \; (d', n') \neq (d, n)$および共同親の$\b{z}\_{d', n'} \; (d', n') \neq (d, n)$であることが分かります．

![](data/2020-02-21-lda-collapsed-gibbs/cond2-2.png)

したがって

\begin{align}
p(\bo{\Phi} | \b{W}\_{d, \bs n}, \b{W}\_{\bs d}, \b{z}\_{d, n}, \b{Z}\_{d, \bs n}, \b{Z}\_{\bs d}) = p(\bo{\Phi} | \b{W}\_{d, \bs n}, \b{W}\_{\bs d}, \b{Z}\_{d, \bs n}, \b{Z}\_{\bs d})
\end{align}

とできます．以上より，上述した周辺化の式が得られました．

---

ここで$p(\bo{\Phi} \| \b{W}\_{d, \bs n}, \b{W}\_{\bs d}, \b{Z}\_{d, \bs n}, \b{Z}\_{\bs d})$を計算していきます．

\begin{align}
p(\bo{\Phi} | \b{W}\_{d, \bs n}, \b{W}\_{\bs d}, \b{Z}\_{d, \bs n}, \b{Z}\_{\bs d})
&\propto p(\bo{\Phi}, \b{W}\_{d, \bs n}, \b{W}\_{\bs d}, \b{Z}\_{d, \bs n}, \b{Z}\_{\bs d}) \newline
&= p(\b{W}\_{d, \bs n}, \b{W}\_{\bs d} | \b{Z}\_{d, \bs n}, \b{Z}\_{\bs d}, \bo{\Phi}) p(\b{Z}\_{d, \bs n}) p(\b{Z}\_{\bs d}) p(\bo{\Phi}) \newline
&\propto p(\b{W}\_{d, \bs n}, \b{W}\_{\bs d} | \b{Z}\_{d, \bs n}, \b{Z}\_{\bs d}, \bo{\Phi}) p(\bo{\Phi}) \newline
&= \left\\{ \prod\_{(d', n') \neq (d, n)} p(\b{w}\_{d', n'} | \b{z}\_{d', n'}, \bo{\Phi}) \right\\} \left\\{ \prod\_{k = 1}^K p(\bo{\phi}_k) \right\\} \newline
&= \left\\{ \prod\_{(d', n') \neq (d, n)} \prod\_{k = 1}^K \mathrm{Cat} (\b{w}\_{d', n'} | \bo{\phi}_k)^{z\_{d', n', k}} \right\\} \left\\{ \prod\_{k = 1}^K \mathrm{Dir} (\bo{\phi}_k | \bo{\beta}) \right\\}
\end{align}

対数をとると

\begin{align}
& \ln p(\bo{\Phi} | \b{W}\_{d, \bs n}, \b{W}\_{\bs d}, \b{Z}\_{d, \bs n}, \b{Z}\_{\bs d}) \newline
&= \sum\_{(d', n') \neq (d, n)} \sum\_{k = 1}^K z\_{d', n', k} \ln \mathrm{Cat} (\b{w}\_{d', n'} | \bo{\phi}_k) + \sum\_{k = 1}^K \ln \mathrm{Dir} (\bo{\phi}_k | \bo{\beta}) + \mathrm{const.} \newline
&= \sum\_{(d', n') \neq (d, n)} \sum\_{k = 1}^K z\_{d', n', k} \sum\_{v = 1}^V w\_{d', n', v} \ln \phi\_{k, v} + \sum\_{k = 1}^K \sum\_{v = 1}^V (\beta_v - 1) \ln \phi\_{k, v} + \mathrm{const.} \newline
&= \sum\_{k = 1}^K \sum\_{v = 1}^V \left( \sum\_{(d', n') \neq (d, n)} z\_{d', n', k} w\_{d', n', v} + \beta_v - 1 \right) \ln \phi\_{k, v} + \mathrm{const.}
\end{align}

となります．よって各$\bo{\phi}_k$毎に独立に分布を求めることができます．

\begin{align}
p(\bo{\Phi} \| \b{W}\_{d, \bs n}, \b{W}\_{\bs d}, \b{Z}\_{d, \bs n}, \b{Z}\_{\bs d})
= \prod\_{k = 1}^K p(\bo{\phi}_k \| \b{W}\_{d, \bs n}, \b{W}\_{\bs d}, \b{Z}\_{d, \bs n}, \b{Z}\_{\bs d})
= \prod\_{k = 1}^K \mathrm{Dir} (\bo{\phi}_k | \tilde{\bo{\beta}}\_{d, \bs n}^{(k)})
\end{align}

\begin{align}
\tilde{\beta}\_{d, \bs n, v}^{(k)} = \sum\_{(d', n') \neq (d, n)} z\_{d', n', k} w\_{d', n', v} + \beta_v
\end{align}

これより

\begin{align}
& p(\b{w}\_{d, n} | \b{W}\_{d, \bs n}, \b{W}\_{\bs d}, \b{z}\_{d, n}, \b{Z}\_{d, \bs n}, \b{Z}\_{\bs d}) \newline
&= \int p(\b{w}\_{d, n} | \b{z}\_{d, n}, \bo{\Phi}) p(\bo{\Phi} | \b{W}\_{d, \bs n}, \b{W}\_{\bs d} \b{Z}\_{d, \bs n}, \b{Z}\_{\bs d}) d \bo{\Phi} \newline
&= \int \left\\{ \prod\_{k = 1}^K \mathrm{Cat} (\b{w}\_{d, n} | \bo{\phi}_k)^{z\_{d, n, k}} \right\\} \left\\{ \prod\_{k = 1}^K \mathrm{Dir} (\bo{\phi}_k | \tilde{\bo{\beta}}\_{d, \bs n}^{(k)}) \right\\} d \bo{\Phi} \newline
&= \int \left\\{ \prod\_{k = 1}^K \mathrm{Cat} (\b{w}\_{d, n} | \bo{\phi}_k)^{z\_{d, n, k}} \mathrm{Dir} (\bo{\phi}_k | \tilde{\bo{\beta}}\_{d, \bs n}^{(k)}) \right\\} d \bo{\Phi} \newline
&= \left( \int \mathrm{Cat} (\b{w}\_{d, n} | \bo{\phi}_1)^{z\_{d, n, 1}} \mathrm{Dir} (\bo{\phi}_1 | \tilde{\bo{\beta}}\_{d, \bs n}^{(1)}) d \bo{\phi}_1 \right) \cdots \left( \int \mathrm{Cat} (\b{w}\_{d, n} | \bo{\phi}_K)^{z\_{d, n, K}} \mathrm{Dir} (\bo{\phi}_K | \tilde{\bo{\beta}}\_{d, \bs n}^{(K)}) d \bo{\phi}_K \right) \newline
\end{align}

と変形できます．ここで$z\_{d, n, k} = 1$のときにどうなるか考えてみます．これは同時に$k' (\neq k)$に対して$z\_{d, n, k'} = 0$を意味するので，$k'$に対応する項を変形してみると

\begin{align}
\int \underbrace{\mathrm{Cat} (\b{w}\_{d, n} | \bo{\phi}\_{k'})^{z\_{d, n, k'}}}_{=1} \mathrm{Dir} (\bo{\phi}\_{k'} | \tilde{\bo{\beta}}\_{d, \bs n}^{(k')}) d \bo{\phi}\_{k'} = \int \mathrm{Dir} (\bo{\phi}\_{k'} | \tilde{\bo{\beta}}\_{d, \bs n}^{(k')}) d \bo{\phi}\_{k'} = 1
\end{align}

となり，$\bo{\phi}\_{k'}$についての確率分布を周辺化しているだけになるので，単純に1になります．したがって

\begin{align}
& p(\b{w}\_{d, n} | \b{W}\_{d, \bs n}, \b{W}\_{\bs d}, z\_{d, n, k} = 1, \b{Z}\_{d, \bs n}, \b{Z}\_{\bs d}) \newline
&= \int \mathrm{Cat} (\b{w}\_{d, n} | \bo{\phi}_k)^{z\_{d, n, k}} \mathrm{Dir} (\bo{\phi}_k | \tilde{\bo{\beta}}\_{d, \bs n}^{(k)}) d \bo{\phi}_k \newline
&= \int \mathrm{Cat} (\b{w}\_{d, n} | \bo{\phi}_k) \mathrm{Dir} (\bo{\phi}_k | \tilde{\bo{\beta}}\_{d, \bs n}^{(k)}) d \bo{\phi}_k \newline
&= \mathrm{Cat} (\b{w}\_{d, n} | \hat{\bo{\beta}}\_{d, \bs n}^{(k)})
\end{align}

\begin{align}
\hat{\beta}\_{d, \bs n, v}^{(k)} \propto \sum\_{(d', n') \neq (d, n)} z\_{d', n', k} w\_{d', n', v} + \beta_v \quad \left( \sum\_{v = 1}^V \hat{\beta}\_{d, \bs n, v}^{(k)} = 1 \right)
\end{align}

が得られます．

さて，ここまでの結果を整理します．まず求めたい分布は

\begin{align}
p(\b{z}\_{d, n} | \b{Z}\_{d, n}, \b{Z}\_{\bs d}, \b{W}) \propto p(\b{w}\_{d, n} | \b{W}\_{d, \bs n}, \b{W}\_{\bs d}, \b{z}\_{d, n}, \b{Z}\_{d, \bs n}, \b{Z}\_{\bs d}) p(\b{z}\_{d, n} | \b{Z}\_{d, \bs n})
\end{align}

でした．$ p(\b{z}\_{d, n} | \b{Z}\_{d, \bs n}) = \mathrm{Cat} (\b{z}\_{d, \bs n} \| \hat{\bo{\alpha}}\_{d, \bs n})$だったことを踏まえると，$z\_{d, n, k} = 1$のとき
\begin{align}
p(z\_{d, n, k} = 1 | \b{Z}\_{d, n}, \b{Z}\_{\bs d}, \b{W}) 
&\propto p(\b{w}\_{d, n} | \b{W}\_{d, \bs n}, \b{W}\_{\bs d}, z\_{d, n, k} = 1, \b{Z}\_{d, \bs n}, \b{Z}\_{\bs d}) p(z\_{d, n, k} = 1 | \b{Z}\_{d, \bs n}) \newline
&= \mathrm{Cat} (\b{w}\_{d, n} | \hat{\bo{\beta}}\_{d, \bs n}^{(k)}) \hat{\alpha}\_{d, \bs n, k}
\end{align}

です．さらに，$p(z\_{d, n, k} = 1 \| \b{Z}\_{d, n}, \b{Z}\_{\bs d}, \b{W})$は条件として$\b{w}\_{d, n}$を含んでいるので，$w\_{d, n, v} = 1$に対応する$\hat{\beta}\_{d, \bs n, v}^{(k)}$が残り，

\begin{align}
p(z\_{d, n, k} = 1 | \b{Z}\_{d, n}, \b{Z}\_{\bs d}, w\_{d, n, v} = 1 \b{W}\_{d, \bs n}, \b{W}\_{\bs d}) \propto \hat{\beta}\_{d, \bs n, v}^{(k)} \hat{\alpha}\_{d, \bs n, k}
\end{align}

となります．あとは$k = 1, \ldots, K$に対して同様に計算し，和が1になるように正規化することで$\b{z}\_{d, n}$についてのカテゴリ分布が得られます．


## References

- [ベイズ推論による機械学習入門](https://www.kspub.co.jp/book/detail/1538320.html)

[bayes-book]: https://www.kspub.co.jp/book/detail/1538320.html
