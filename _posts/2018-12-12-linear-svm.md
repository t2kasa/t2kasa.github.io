---
title:  "線形SVM"
permalink: 2018-12-31-linear-svm.html
sidebar: blog_sidebar
tags: []
---

SVMを復習しよう．ふとそう思った．今回は最も基本的なケースであるハードマージンかつ線形のSVMについて書く．これは2クラス分類問題において，ある線形判別関数$g(\bx)$によって線形分離可能であることを仮定している．

### CVXOPTによる実装例

凸最適化のためのパッケージ[CVXOPT](https://github.com/cvxopt/cvxopt)で$\balpha$を求めてみよう．
