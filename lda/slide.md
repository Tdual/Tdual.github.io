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
## LDAとは
## LDAはLatent Dirichlet Allocationの略
日本語では「潜在ディリクレ分配法」
- 論文：David M. Blei, Andrew Y. Ng,Michael I. Jordan, _Latent Dirichlet Allocation_
 http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf

- 文章ごとのトピック分布が「ディリクレ分布」に従う
- ベイズ推定を使う
$$Dir_k(\theta|\alpha)=\frac{\Gamma \left(\sum_i^k \alpha_i \right)}{\prod^k_i\Gamma(\alpha_i)}\prod_i^k\theta_i^{\alpha_i - 1}$$
※数式が崩れているときはブラウザのリロード

---
## LDA その２
### 文章が従う分布
今まで話したことを数式で表現すると
$$p(D| \alpha, \eta) = \prod_d^M p\left(w_d |\alpha,\eta \right)= \prod_d^M \int \int Dir_k(\theta_d|\alpha) \left(\prod^{N_d}_n \sum_z^k p(z_d^n|\theta_d)
p\left(w_d^n| z_d^n,\beta \right) \right)p(\beta| \eta) d\theta_d d\beta$$
<center><img src=lda.png width=50%></center>
---

## LDA その３
#### 適切なハイパーパラメータα,ηを求めたい
周辺尤度を最大化だ！
$$p\left(D | \alpha,\eta \right)
\equiv \prod_d^M\prod^{N_d}_n\sum_z^k \int \int p\left(w_d^n,z_d^n,\theta, \beta | \alpha,\eta \right)d\theta d\beta$$

このままでは積分計算できないのでｐに近い分布qを求めることを考える
$$p\left(D | \alpha,\eta \right)
=\prod_d^M\prod^{N_d}_n\sum_z^k \int \int q\left(z,\theta,\beta | \gamma, \phi \right)\frac{p\left(w,z,\theta, \beta | \alpha,\eta \right)}{q(z,\theta,\beta | \gamma, \phi)}d\theta d\beta$$

まだqも扱いづらいので独立分布に近似する（平均場近似）
$$q(z,\theta,\beta | \gamma, \phi) = q(z)q(\theta,\beta | \gamma,\phi) \equiv q(z)q(\theta,\beta)$$

$$\sum_z q(z) = 1,\int\int q(\theta,\beta) d\theta d\beta = 1$$
---
## LDA その４
#### 適切なハイパーパラメータα,ηを求めたい
logとると都合がいい（桁落ち防止、単調増加でなめらか凸関数）のでlogとって
$$\log p\left(D,\theta,\beta | \gamma, \phi \right)
= \log \left(\prod_d^M\prod^{N_d}_n\sum_z^k \int \int q(z)q(\theta)q(\beta)\frac{p\left(w,z,\theta, \beta |\alpha,\eta \right)}{q(z)q(\theta,\beta)}d\theta d\beta \right)$$
$$\geq \prod_d^M\prod^{N_d}_n\sum_z^k \int \int q(z)q(\theta,\beta) \log \frac{p\left(w,z,\theta, \beta | \alpha,\eta \right)}{q(z)q(\theta,\beta)}d\theta d\beta
\equiv \prod_d^M\prod^{N_d}_nI\left(q(z),q(\theta,\beta)\right)$$

Iを最大化すするようなq(z),q(θ,β)を求めたい！

ちなみに、二行目はJensenの不等式を使った。f(x)が上に凸の時
$$f\left( \int y(x)p(x) dx \right) \ge \int f(y(x))p(x) dx$$


---
## 変分法 その1
#### 極値（停留点）のパラメータを求める方法

とりあえず、被積分関数をLと置く。
$$I\left(q(z),q(\theta,\beta)\right) = \sum_z\int \int L\left(q(z),q(\theta,\beta)\right) d\theta d\beta$$

停留点ならばδｑを少し動かしてもIの変化は０のはず。

$$\delta I = I\left(q(z)+\delta q(z),q(\theta,\beta)+\delta q(\theta,\beta)\right) - I\left(q(z),q(\theta),q(\beta)\right) = 0$$
$$\sum_z\int \int\left(L \left(q(z)+\delta q(z),q(\theta,\beta)+\delta q(\theta,\beta)\right) - L(q(z),q(\theta,\beta))\right)d \theta d\beta=0$$
被積分関数を１次までTaylor展開して
$$\sum_z\int\int \left(L \left(q(z),q(\theta,\beta) \right) + \frac{\partial L}{\partial q(z)}\delta q(z)+ \frac{\partial L}{\partial q(\theta,\beta)}\delta q(\theta,\beta) - L \left(q(z),q(\theta),q(\beta) \right) \right)d \theta d \beta=0$$

---
### 変分法 その２
$$\sum_z\int\int\left(\frac{\partial L}{\partial q(z)}\delta q(z)+\frac{\partial L}{\partial q(\theta,\beta)}\delta q(\theta,\beta) \right)d\theta d\beta=0$$
$$\sum_z\left(\int\int\frac{\partial L}{\partial q(z)}d\theta\beta\right)\delta q(z)
+\int\int \left(\sum_z\frac{\partial L}{\partial q(\theta,\beta)} \right)\delta q(\theta,\beta)d\theta d\beta  = 0$$

 任意のδq(z),q(θ,β)について成り立つのでそれぞれの項で被積分関数が０しか有り得ない。

δq(θ,β)方向の変分が満たす式は
$$\sum_z\frac{\partial L}{\partial q(\theta,\beta)} = 0$$

δq(z)方向の変分は
$$\int\int\frac{\partial L}{\partial q(z)}d\theta d\beta = 0$$
※パラメータ微分関数がない場合のEuler-Lagrange方程式に他ならない


---
## LDA その5
#### 適切なハイパーパラメータα,ηを求めたい

$$L = q(z)q(\theta,\beta) \log \frac{p\left(w,z,\theta, \beta | \alpha,\eta \right)}{q(z)q(\theta,\beta)}
=q(z)q(\theta,\beta)(\log p\left(w,z,\theta, \beta | \alpha,\eta \right) - \log q(z) - \log q(\theta,\beta) )$$

δq(θ,β)方向の変分を求める。
$$\sum_z\frac{\partial L}{\partial q(\theta,\beta)}
= \sum_z \left(q(z) \left(\log p\left(w,z,\theta, \beta | \alpha,\eta \right) - \log q(\theta,\beta) \right) - q(z)q(\theta,\beta)\frac{1}{q(\theta,\beta)} \right)
=0$$

$$q(\theta) = Ce^{\sum q(z)\log p\left(w,z,\theta, \beta | \alpha,\eta \right) }=Ce^{\left<\log p\left(w,z,\theta, \beta | \alpha,\eta \right)\right>_{q(z)}}(Cは定数)$$

同様にδq(z)方向の変分を求めると。

$$q(z) = Ce^{\left<\log  p\left(w,z,\theta, \beta | \alpha,\eta \right) \right>_{q(\theta,\beta)}}$$

---
## LDA その6
#### 適切なハイパーパラメータα,ηを求めたい

$$p\left(w,\theta, \beta | \alpha,\eta \right)=
Dir_k(\theta_d|\alpha) \left(\prod^{N_d}_n \sum_z^k p(z_d^n|\theta_d)p\left(w_d^n| z_d^n,\beta \right)\right)p(\beta| \eta)$$

$$p(\beta|\eta) = Dir_k(\beta|\eta)$$
他の分布を多項分布として計算すると

$$q(\theta_d) = Dir_k(\theta_d|\alpha_d^k),q(\beta_d^i)=Dir_k(\beta_d^i|\eta_d^{iw})$$
$$\alpha_d^k = \alpha_d + \sum_n^{N_d}r(w_d^n)^k,
\eta_d^{iw} = \eta_d^{i} + \sum_n^{N_d}r(w_d^n)^i$$
$$r(w_d^n) = e^{\Psi(\alpha^k_d)-\Psi(\sum_k\alpha^k_d)+\Psi(\eta_w^k)-\Psi(\sum_v\eta^k_v)}$$
->αとηの関係式がが求まった。初期値として適当なα,ηを選んでrを導出してrでα,ηを更新を繰り返すアルゴリズムを組めば良いだけ。


---
## gensim

### トピックモデリングのためのPythonライブラリ
<center><img src=gensim.png width=70%></center>
https://radimrehurek.com/gensim/

---
## gensimでLDA

例：
https://github.com/Tdual/topic_model/blob/master/LDA_gensim.ipynb

---
## matrix factrization的な解釈

$$p\left(w_d |\alpha,\eta \right)
=\int \int Dir_k(\theta_d|\alpha) \left(\prod^{N_d}_n \sum_z^k p(z_d^n|\theta_d)
p\left(w_d^n| z_d^n,\beta \right)p(\beta| \eta) \right)d\theta_d d\beta$$
文章の生成確率の式をよく見ると、θ、ηを与えた時にd番目の文章のn番目の単語の出現確率が含まれる。
$$p(w_d^n|\theta,\eta)
\equiv \sum_z^k p(z_d^n|\theta_d)p\left(w_d^n| z_d^n,\beta \right)
\equiv \sum_i^k \theta_i^d \beta_i^{w}=\Theta B $$

$$(z_d^n \equiv i, p(i|\theta_d) \equiv \theta_i^d, p\left(w_d^n| i,\beta \right) \equiv \beta_i^{w})$$

Θは（M,i）行列,Bは(i,V)行列,（Mは文章数、Vは単語数）
<center><img src=factrization.png width=50%></center>
