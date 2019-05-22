# Domain Generalization by Solving Jigsaw Puzzles (CVPR2019)

## 概要

- ジグソーパズルを解くself-supervised learningをdomain generalizationに応用．

## 問題設定

- $S$個のsource domainがラベル付きで与えられる．
- 未知のtarget domianで評価する．
- $i$番目のsource domainは$N_i$個のラベル付きのインスタンス集合$$がある．

## 損失

まず，sourceの場合，訓練する場合のシナリオについて述べる．
クラス分類タスクを想定．損失は以下の2つからなる．
- クラス分類の損失 cross-entropy loss
- ジグソーパズルのpermutation indexを当てる（損失はクラス分類と同様に計算できる）

[eq. 1]

[Fig 2]

次に，targetの場合は
- クラス分類の損失はclassificationの結果をそのまま用いる． empirical entropy loss
- ジグソーパズルの損失はsourceと同様に求まるのでそのまま．

[eq in sentence]

## 評価

モデルを評価する際は，ジグソーパズルのpermutattion indexをclassificationする箇所は用いず，
画像のクラス分類の部分のみ用いる．

## この後読むべき論文

- self-supervised learning related papers
    - [36] Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles (ECCV2016) [Arxiv](https://arxiv.org/abs/1603.09246)
    - [38]
