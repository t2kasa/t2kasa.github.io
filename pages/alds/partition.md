---
title: 分割数
tags: [アルゴリズムとデータ構造]
sidebar: home_sidebar
permalink: partition.html
---

**ほぼ全面的にこちらの記事を参考にさせて頂いたのでこちらをご覧ください．**  
http://d.hatena.ne.jp/incognita/20110305/1299344781

### 定義
$P(n, m) :=$順序を区別せずに整数$n$を$m$以下の自然数の和に分ける場合の数

### 初期条件と漸化式
\begin{align}
&P(0, m) = 1 \newline
&P(n, 1) = 1 \newline
&P(n, m) = P(n - m, m) + P(n, m - 1) \quad (n \geq m > 1) \newline
&P(n, m) = P(n, n) \quad (n < m)
\end{align}

### 考え方
0以上の整数$n$の分割数$P(n)$は「順序を区別せずに$n$を自然数の和に分ける場合の数」． 
ただし，$P(0) = 1$としておく．

例：$P(7) = 15$
```
7
6 + 1
5 + 2
5 + 1 + 1
4 + 3
4 + 2 + 1
4 + 1 + 1 + 1
3 + 3 + 1
3 + 2 + 2
3 + 2 + 1 + 1
3 + 1 + 1 + 1
2 + 2 + 2 + 1
2 + 2 + 1 + 1 + 1
2 + 1 + 1 + 1 + 1 + 1
1 + 1 + 1 + 1 + 1 + 1 + 1
```

最大の数が$3$のグループの組み合わせ数は先頭の$3$を取り除いた残りの$4 (= 7 - 3)$を3以下の自然数に分割する組み合わせ数に一致する．即ち$P(7, 3) = P(7 - 3, 3) = P(4, 3)$

「7を3以下の自然数に分割する場合の数」 = 「最大の数が3のグループの組み合わせ数」 +「最大の数が2のグループの組み合わせ数」 + 「最大の数が1のグループの組み合わせ数」になっている．  
最大の数が3のグループの組み合わせ数は$P(7 - 3, 3)$であった．  
したがって  
「7を3以下の自然数の和に分割する場合の数」 = 「最大の数が3のグループの組み合わせ数」 + 「7を2以下の自然数の和に分割する場合の数」  
$P(7, 3) = P(4, 3) + P(7, 2)$  
よって$P(n, m) = P(n - m, m) + P(n, m - 1) \quad (n \geq m)$を得る．

### 参考

* http://d.hatena.ne.jp/incognita/20110305/1299344781