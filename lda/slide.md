class: center, middle
# Topic model & LDA & gensims

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
# LDA その１
## LDAはLatent Dirichlet Allocationの略
## 日本語では「潜在ディリクレ分配法」
### ・文章ごとのトピック分布が「ディリクレ分布」に従う
### ・ベイズ推定を使う
$$Dir(\theta|\alpha)=\frac{\Gamma \left(\sum_i^k \alpha_i \right)}{\prod^k_i\Gamma(\alpha_i)}\prod_i^k\theta_i^{\alpha_i - 1}$$

---
# LDA その２
### 文章が従う分布
今まで話したことを数式で表現すると
$$p\left(w |\alpha,\eta \right)= \int \int Dir(\theta|\alpha) \left(\prod^N_n \sum_z p(z_n|\theta)p(w_n|z_n,\beta)p(\beta| \eta) \right)d\theta d\beta$$
<center><img src=lda.png width=50%></center>
---

# LDA その３
### 適切なハイパーパラメータα,ηを求めたい
尤度を最大化だ！
$$p\left(w | \alpha,\eta \right)= \int \int Dir(\theta|\alpha) \left(\prod^N_n \sum_z p(z_n|\theta)p(w_n|z_n,\beta)p(\beta| \eta) \right)d\theta d\beta=\int \int p\left(w,\theta, \beta | \alpha,\eta \right)d\theta d\beta$$

このままでは積分計算できないのでｐに近い分布qを求めることを考える
$$p\left(w | \alpha,\eta \right)=\int \int q\left(\theta,\beta | \gamma, \phi \right)\frac{p\left(w,\theta, \beta | \alpha,\eta \right)}{q(\theta,\beta | \gamma, \phi)}d\theta d\beta$$

まだqも扱いづらいので独立分布に近似する（平均場近似）
$$q(\theta,\beta | \gamma, \phi) = q\left(\theta | \gamma \right)q\left(\beta |\phi \right) \equiv q(\theta)q(\beta)$$

$$\int q(\theta) d\theta = 1, \int q(\beta) d\beta = 1$$

---
# LDA その４
### 適切なハイパーパラメータα,ηを求めたい
logとると都合がいい（桁落ち防止、単調増加でなめらか凸関数）のでlogとって
$$\log p\left(w,\theta,\beta | \gamma, \phi \right) = \log \int \int q(\theta)q(\beta)\frac{p\left(w,\theta, \beta |\alpha,\eta \right)}{q(\theta)q(\beta)}d\theta d\beta$$
$$\geq \int \int q(\theta)q(\beta) \log \frac{p\left(w,\theta, \beta | \alpha,\eta \right)}{q(\theta)q(\beta)}d\theta d\beta
\equiv I\left(q(\theta),q(\beta)\right)$$

Iを最大化すするような

Jensenの不等式を使った。
$$\int f(y(x))p(x) dx \ge f\left( \int y(x)p(x) dx \right)$$


---
# 変分法 その1
### 極値（停留点）の時のパラメータを求める方法

$$I\left(q(\theta),q(\beta)\right) = \int \int L\left(q(\theta),q(\beta)\right) d\theta d\beta$$

停留点ならばδｑを少し動かしてもIの変化は０のはず。
$$\delta I = 0$$

$$\delta I = I\left(q(\theta)+\delta q(\theta)+\delta q(\beta)\right) - I(q(\theta),q(\beta))$$
$$=\int \int[L \left(q(\theta)+\delta q(\theta) +\delta q(\beta) \right) - L(q(\theta),q(\beta))]d \theta d\beta$$
$$=\int\int [L \left(q(\theta) \right) + \frac{\partial L}{\partial q(\theta)}\delta q(\theta) + \frac{\partial L}{\partial q(\beta)}\delta q(\beta) - L\left(q(\theta)\right)] d \theta d \beta$$

---
# 変分法 その２
### 極値（停留点）の時のパラメータを求める方法
$$= \int\int[\frac{\partial L}{\partial q(\theta)}\delta q(\theta) + \frac{\partial L}{\partial q(\beta)}\delta q(\beta)]d\theta d\beta$$
$$=\int(\int\frac{\partial L}{\partial q(\theta)} d\beta)\delta q(\theta)d\theta + \int(\int\frac{\partial L}{\partial q(\beta)} d\theta)\delta q(\beta)d\beta = 0$$

 任意のδq(θ),δq(θ)について成り立つのでそれぞれの項で被積分関数が０しか有り得ない。

δq(θ)方向の変分は

$$\int\frac{\partial L}{\partial q(\theta)} d\beta = 0$$

δq(β)方向の変分は

$$\int\frac{\partial L}{\partial q(\beta)} d\theta = 0$$

---
# LDA その5
### 適切なハイパーパラメータα,ηを求めたい

$$L = q(\theta)q(\beta) \log \frac{p\left(w,\theta, \beta | \alpha,\eta \right)}{q(\theta)q(\beta)}
=q(\theta)q(\beta)(\log p\left(w,\theta, \beta | \alpha,\eta \right) - \log q(\theta) - \log q(\beta))$$

δq(θ)方向の変分を求める。
$$\int\frac{\partial L}{\partial q(\theta)} d\beta
= \int \left[q(\beta) \left(\log p\left(w,\theta, \beta | \alpha,\eta \right) - \log q(\theta) - \log q(\beta) \right) - q(\theta)q(\beta)\frac{1}{q(\theta)} \right]d\beta=0$$

$$q(\theta) = Ce^{\int q(\beta)\log p\left(w,\theta, \beta | \alpha,\eta \right) d\beta}=Ce^{<\log  p\left(w,\theta, \beta | \alpha,\eta \right)>_{q(\beta)}}(Cは定数)$$

同様にδq(β)方向の変分を求める。
$$q(\beta) = Ce^{<\log  p\left(w,\theta, \beta | \alpha,\eta \right)>_{q(\theta)}}$$

---
# LDA その6
### 適切なハイパーパラメータα,ηを求めたい

$$q(\theta) = Ce^{<\log  p\left(w,\theta, \beta | \alpha,\eta \right)>_{q(\beta)}}$$

$$q(\beta) = Ce^{<\log  p\left(w,\theta, \beta | \alpha,\eta \right)>_{q(\theta)}}$$

あとは、相互最適化の問題

初期値として適当なα、ηとq（θ)を与えてIを最大にするq(β)を求める。(積分計算が難しかっただけでpは既知なので。)  
そのq(β）からIを最大にするq(θ)、α、ηを求める。  
収束するまで繰り返す  

### ->最適なα、ηが求まる。  

### そういうアルゴリズムを組めば良いだけ。
### ね？簡単でしょう？



---
# gensim

### トピックモデリングのためのPythonライブラリ
<center><img src=gensim.png width=70%></center>
https://radimrehurek.com/gensim/

---
# gensimでLDA
インストール
```
pip install gensim
```
例：
https://github.com/Tdual/topic_model/blob/master/LDA_gensim.ipynb
