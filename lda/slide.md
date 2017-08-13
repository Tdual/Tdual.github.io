class: center, middle
# Topic model & LDA & gensims

---

# Agenda

1. Introduction
2. Deep-dive
3. ...

---

# トピックモデル
### データが”複数”の「トピック」から確率的に生成されると仮定したモデル
### 自然言語処理の世界ではこのデータは文章のこと。
#### (このスライドでは以降は自然言語処理の文脈で話します。)



---

# トピック
### データ（文章）の背後に隠れた潜在的な意味。単語の分布  
<center><img src=topic.png width=90%></center>
---

# 文章
<center><img src=doc.png width=83%></center>

---
# LDA（1）
## LDAはLatent Dirichlet Allocationの略
## 日本語では「潜在ディリクレ分配法」
### ・文章ごとのトピック分布が「ディリクレ分布」に従う
### ・ベイズ推定を使う
$$Dir(\theta|\alpha)=\frac{\Gamma \left(\sum_i^k \alpha_i \right)}{\prod^k_i\Gamma(\alpha_i)}\prod_i^k\theta_i^{\alpha_i - 1}$$

---
# LDA (2)
### 文章が従う分布
$$p(w|\alpha,\beta)= \int Dir(\theta|\alpha) \left(\prod^N_n \sum_z p(z_n|\theta)p(w_n|z_n,\beta) \right)d\theta$$
---

# LDA
$$ i\gamma^{\mu}\partial_{\mu}\psi -m\psi = 0$$
