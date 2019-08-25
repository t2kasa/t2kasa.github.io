---
title: "Adversarial Variational Bayes: Unifying Variational Autoencoders and Generative Adversarial Networks (ICML2017)"
sidebar: home_sidebar
permalink: avb.html
---

## VAE

まず，通常のVAEの問題設定を確認しよう．parametric generative model $p_\theta (x | z)$とinference model $q_{\phi} (z | x)$を用いて対数尤度$\log p_\theta (x)$を下からevidence lower bound (ELBO)でバウンドする．
$$
\log p_\theta (x) \geq - \mathrm{KL}( q_{\phi} (z | x) \| p(z)) + \mathbb{E}_{q_{\phi} (z | x)} \left[ \log p_{\theta} (x | z) \right]
$$
そしてdata distribution $p_{\mathcal{D}} (x)$の期待値の下で最大化する問題として扱う．
$$
\max_{\theta, \phi} \mathbb{E}_{p_{\mathcal{D}} (x)} \left[ - \mathrm{KL}( q_{\phi} (z | x) \| p(z)) + \mathbb{E}_{q_{\phi} (z | x)} \left[ \log p_{\theta} (x | z) \right] \right]
$$

## Adversarial Variational Bayes (AVB)

まず上式をもう少し変形する．
$$
\max_{\theta, \phi} \mathbb{E}_{p_{\mathcal{D}} (x)} \left[ - \mathrm{KL}( q_{\phi} (z | x) \| p(z)) + \mathbb{E}_{q_{\phi} (z | x)} \left[ \log p_{\theta} (x | z) \right] \right]
= \max_{\theta, \phi} \mathbb{E}_{p_{\mathcal{D}} (x)} \mathbb{E}_{q_{\phi} (z | x)} \left[ \log p(z) - \log q_{\phi} (z | x) + \log p_{\theta} (x | z) \right]
$$
$q_\phi (z | x)$がガウス分布の場合等は，VAEで扱ったようにreparameterization trickによってSGDによって最適化できる．問題はそのように明示的な分布が分かっていない場合である．

**AVBのアイディアは$\log p(z) - \log q_\phi (z | x)$を最適値としてとるようなネットワーク$T(x, z)$を導入して，これらの項を置き換えてしまう．**

このような$T(x, z)$は，$T(x, z)$をdiscriminatorとして，固定された$q_\phi (z | x)$に対して以下の最適化問題を解けばよい．
$$
\max_T \mathbb{E}_{p_{\mathcal{D}} (x)} \mathbb{E}_{q_{\phi} (z | x)} \left[ \log \sigma (T(x, z)) \right] + \mathbb{E}_{p_{\mathcal{D}} (x)} \mathbb{E}_{p(z)} \left[ 1 - \sigma(T(x, z)) \right]
$$
この最適解は$T^*(x, z) = \log q_\phi (z | x) - \log p(z)$になる．よって，最初の最適化問題は次のように書ける．
$$
\max_{\theta, \phi} \mathbb{E}_{p_{\mathcal{D}} (x)} \mathbb{E}_{q_{\phi} (z | x)} \left[ - T^* (x, z) + \log p_{\theta} (x | z) \right]
$$
さらにreparameterization trickによって
$$
\max_{\theta, \phi} \mathbb{E}_{p_{\mathcal{D}} (x)} \mathbb{E}_{\epsilon} \left[ - T^* (x, z_\phi (x, \epsilon)) + \log p_{\theta} (x | z_\phi (x, \epsilon)) \right]
$$
を得ることができる．

しかし，常に$T^*(x, z)$を求めるのは困難である．そこで，実際のtrainingでは以下の2つの式をtwo-player gameとして最適化する．
$$
\max_{\theta, \phi} \mathbb{E}_{p_{\mathcal{D}} (x)} \mathbb{E}_{\epsilon} \left[ - T (x, z_\phi (x, \epsilon)) + \log p_{\theta} (x | z_\phi (x, \epsilon)) \right] \\
\max_T \mathbb{E}_{p_{\mathcal{D}} (x)} \mathbb{E}_{q_{\phi} (z | x)} \left[ \log \sigma (T(x, z)) \right] + \mathbb{E}_{p_{\mathcal{D}} (x)} \mathbb{E}_{p(z)} \left[ 1 - \sigma(T(x, z)) \right]
$$

## References

- Adversarial Variational Bayes: Unifying Variational Autoencoders and Generative Adversarial Networks
    - http://proceedings.mlr.press/v70/mescheder17a.html
    - https://github.com/LMescheder/AdversarialVariationalBayes
