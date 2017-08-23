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
今まで話したことを数式で表現すると
$$p\left(w |\alpha,\eta \right)= \int \int Dir(\theta|\alpha) \left(\prod^N_n \sum_z p(z_n|\theta)p(w_n|z_n,\beta)p(\beta| \eta) \right)d\theta d\beta$$
<center><img src=lda.png width=50%></center>
---

# LDA
### 適切なハイパーパラメータα,ηを求めたい
尤度を最大化だ！
$$p\left(w | \alpha,\eta \right)= \int \int Dir(\theta|\alpha) \left(\prod^N_n \sum_z p(z_n|\theta)p(w_n|z_n,\beta)p(\beta| \eta) \right)d\theta d\beta=\int \int p\left(w,\theta, \beta | \alpha,\eta \right)d\theta d\beta$$

このままでは計算できないのでｐに近い分布qを求めることを考える
$$p\left(w | \alpha,\eta \right)=\int \int q\left(\theta,\beta | \gamma, \phi \right)\frac{p\left(w,\theta, \beta | \alpha,\eta \right)}{q(\theta,\beta | \gamma, \phi)}d\theta d\beta$$

まだqも扱いづらいので独立分布に近似する（平均場近似）
$$q(\theta,\beta | \gamma, \phi) = q\left(\theta | \gamma \right)q\left(\beta |\phi \right) \equiv q(\theta)q(\beta)$$

$$\int q(\theta) d\theta = 1, \int q(\beta) d\beta = 1$$

---
# LDA
### 適切なハイパーパラメータα,ηを求めたい
logとると都合がいい（桁落ち防止、単調増加でなめらか凸関数）のでlogとって
$$\log p\left(w,\theta,\beta | \gamma, \phi \right) = \log \int \int q(\theta)q(\beta)\frac{p\left(w,\theta, \beta |\alpha,\eta \right)}{q(\theta)q(\beta)}d\theta d\beta$$
$$\geq \int \int q(\theta)q(\beta) \log \frac{p\left(w,\theta, \beta | \alpha,\eta \right)}{q(\theta)q(\beta)}d\theta d\beta
\equiv I\left(q(\theta),q(\beta)\right)$$

Iを最大化すれば良い

Jensenの不等式
