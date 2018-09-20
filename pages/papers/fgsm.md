---
title: "Explaining and Harnessing Adversarial Examples"
last_updated: Sep 20, 2018
tags: [Adversarial Example, Work In Progress, ICLR2015]
sidebar: home_sidebar
permalink: fgsm.html
---

Adversarial Example関連の論文を全く読んでいなかったので，パンダが例として出てくる論文を眺めることにした．
読めていない箇所もかなりあるが，FGSMの概要は把握できたと思うのでメモとして残しておく．

## 概要

本論文では，損失関数の入力変数についての勾配を符号を利用する
**Fast Gradient Sign Method (FGSM)**を提案している．
また，FGSMの提案の過程で，単純な線形モデルであってもadversarial exampleを
引き起こすことが可能であることが説明されている．

## 線形モデルにおけるAdversarial Examples

\begin{equation}
\newcommand{\bfx}{\boldsymbol{x}}
\newcommand{\bfw}{\boldsymbol{w}}
\newcommand{\bfeta}{\boldsymbol{\eta}}
\newcommand{\bftheta}{\boldsymbol{\theta}}
\end{equation}

まず，単純な線形モデルでadversarial exampleがどうやって引き起こされるか，について考える．
例えば，画像は基本的に1 channelに対して8bitで表現されるため，ダイナミックレンジの1 / 255
未満の変化は精度の問題から切り捨てられている（補足：センサが取りうる値の範囲がダイナミックレンジ）．ここで，精度によって切り捨てられるほど小さい摂動$\bfeta$を入力$\bfx$に加えたadversarial input$\tilde{\bfx} = \bfx + \bfeta$を考える．ただし$\bfeta$は
精度の問題で切り捨てられるほど十分に小さな値$\epsilon$に対して$\|\bfeta\|_{\infty} = \max{\bfeta} < \epsilon$とする．このとき重みベクトル$\bfw$との内積は
\begin{align}
    \bfw^T \tilde{\bfx} = \bfw^T \bfx + \bfw^T \bfeta
\end{align}
である．すると内積は$\bfeta$を加えたことにより，$\bfw^T \bfeta$だけ変化させることができる．
$\bfeta$のmax norm constraint下において，上式を最大化するような$\bfeta$は$\bfeta = \mathrm{sign}(\bfw)$である．例えば，$\bfw$が$n$次元ベクトルであり，L2ノルムの要素数に対する平均を$m$とすると，結果として$\bfeta$を加えることで$\epsilon m n$変化させることができる．

よって，重みベクトルの次元数$n$に対して線形に大きくなる．
以上の考えから，線形モデルであっても十分に次元数が大きければ，摂動$\bfeta$によって識別結果を
変化させるようなadversarial input$\tilde{\bfx}$を生成することが可能だと考えられる．

## Fast Gradient Sign Method (FGSM)

neural network (NN)は「線形的すぎる」ので上記の線形モデルにおけるlinear adversarial perturbation
に対抗できないだろう，と著者らは仮説を立てている．
LSTMやReLUなどは意図的に線形に振る舞うように設計されており，またsigmoidでもsaturateしていない線形の領域にとどまるように注意深くチューニングされている．
このような線形の振る舞いから、NNは線形モデルにおける摂動が影響を与えられることを示唆している．

NNのモデルのパラメータを$\bftheta$，入力を$\bfx$，出力を$y$に対応する出力とする．
損失関数を$J(\bftheta, \bfx, y)$とすると，max norm constraint下で最適な摂動は
\begin{align}
    \bfeta = \epsilon \mathrm{sign}(\nabla_{\bfx} J(\bftheta, \bfx, y))
\end{align}
となる．以上の手法をFast Gradient Sign Method (FGSM)と呼ぶことにする．
勾配はbackpropによって効率的に求めることが可能である．
