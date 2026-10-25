# Mathematics

[Overview](../README.md) · [Models and training](models-and-training.md)

We construct molecular generative processes by separating shared fragment motion from internal atomic motion, and learn their generators on the observed coordinates and atom labels. An atom-wise Brownian bridge provides the flow-matching baseline. This page develops the conditional processes and training objectives; architecture and training settings are given in [Models and training](models-and-training.md).

## Observed state and conditioning

A molecule has atom coordinates $`x=(x_i)_{i=1}^n`$, atom labels $`a=(a_i)_{i=1}^n`$, and a fixed fragment partition $`\mathcal F=\{I_f\}_{f=1}^m`$, with $`n_f=|I_f|`$. Its observed generative state is $`X_t=(x_t,a_t)`$. Both models condition on $`\mathcal F`$, including the atom count and the membership of every atom. The clean coordinates and atom labels are training endpoints, not network inputs.

Coordinates lie in the molecule-centered subspace. Centering uses the **arithmetic mean of all atom positions**, including hydrogens, rather than atomic masses. Write

```math
(Pz)_i=z_i-\frac1n\sum_jz_j,
\qquad x_0=\sigma_0P\epsilon,
\qquad \epsilon_i\sim\mathcal N(0,I_3).
```

All labels initially equal a mask token. The source standard deviation is $`\sigma_0=3`$ Å before centering; centering gives per-coordinate variance $`\sigma_0^2(1-1/n)`$. Coordinates of different atoms are therefore correlated: $`\mathrm{Cov}(x_{i,0},x_{j,0})=\sigma_0^2(\delta_{ij}-1/n)I_3`$.

The clean endpoint is centered with the same projection. Both conditional processes remain in $`\sum_i x_i=0`$, and both learned samplers recenter after every coordinate update. This fixes the unmodelled global translation; it does not introduce a force drawing each fragment independently toward the origin. In Fragment Noise, fragment-centroid increments and noise are projected using atom-count weights, while internal coordinates and noise have zero mean within each fragment. Brownian Atoms projects its Cartesian noise at the molecule level. The network's translation symmetry and the precise sampling projections are described under [centering and coordinate updates](models-and-training.md#centering-and-coordinate-updates).

## LPGM formulation

Let $`Y_t`$ be the latent Markov process, with generator $`W_t`$, and let $`X_t=\Phi(Y_t)`$. Static endpoints can be included as constant components of $`Y_t`$. For a test function $`f`$, the latent process conditional generator and its marginal are

```math
L_t^{y}f(\Phi(y))=W_t(f\circ\Phi)(y),
\qquad
L_tf(X)=\mathbb E[L_t^{Y_t}f(X)\mid\Phi(Y_t)=X].
```

Conditioning on the supplied partition and fixed process metadata is implicit. Under the regularity and uniqueness assumptions for the pushforward generator, $`L_t`$ generates the time marginals $`p_t^{\mathcal X}=\Phi_\#p_t^{\mathcal Y}`$. This is a statement about one-time distributions, not equality of entire trajectory laws.

For a linear parametrization $`L_t^yf(X)=\langle\mathcal K_{t,X}f,F_t^y(X)\rangle`$, the marginal target is $`F_t(X)=\mathbb E[F_t^{Y_t}(X)\mid X_t=X]`$. Conditional regression with the appropriate Bregman divergence has the same parameter gradient as regression against this marginal target. Here we use quadratic losses for coordinate targets and cross-entropy for endpoint labels. The general results and proofs are given in [the paper](https://arxiv.org/abs/2605.20547).

## Fragment latent process and pushforward

For fragment $`f`$, introduce a centroid $`c_f\in\mathbb R^3`$, a rotation $`R_f\in\mathrm{SO}(3)`$, and centered local coordinates $`q_i\in\mathbb R^3`$ with $`\sum_{i\in I_f}q_i=0`$. The molecular pushforward is

```math
x_i=\Phi_i(Y)=c_f+R_fq_i,\qquad i\in I_f.
```

Thus $`Y_t`$ contains $`(c_{f,t},R_{f,t},q_{i,t},a_{i,t})`$ and the fixed endpoints. The observed coordinates do not specify a unique latent rotation/local-coordinate decomposition.

At each endpoint, $`c_f`$ is the mean of its atoms. A template-aligned frame defines $`R_{f,1}`$ and $`q_{i,1}`$. A random relative rotation defines $`R_{f,0}`$, and $`q_{i,0}=R_{f,0}^{\mathsf T}(x_{i,0}-c_{f,0})`$. Consequently the latent source maps back to exactly the sampled $`x_0`$: there is no separately sampled centroid source.

### Clocks

Centroids and rotations use a common clock $`s(t)`$, while internal coordinates use $`u_q(t)`$:

```math
s(t)=10t^3-15t^4+6t^5,\qquad
u_q(t)=1-[1-s(t)]^p,\qquad
\lambda(t)=\frac{s'(t)}{1-s(t)}.
```

Since $`u_q'/(1-u_q)=p\lambda`$, internal coordinates approach their local endpoints $`p`$ times faster in normalized bridge time. The reference models use $`p=16`$.

### Centroid bridge

Let $`P_c`$ subtract the atom-count-weighted mean of the fragment centroids, so that $`\sum_fn_fc_f=0`$. Define

```math
h(t)=1-\exp[-(t/\tau)^2],\qquad
V_c(t)=\nu_cs(t)[1-s(t)]+B^2[1-s(t)]^2h(t).
```

Conditional on the endpoints, the centroid mean is $`(1-s)c_0+sc_1`$, and its noise covariance is $`V_c(t)P_cP_c^{\mathsf T}\otimes I_3`$. Equivalently,

```math
dc_t=\lambda(t)(c_1-c_t)\,dt+g_c(t)P_c\,dW_t^c,
\qquad
g_c^2(t)=\nu_cs'(t)+B^2[1-s(t)]^2h'(t).
```

The additional term spreads centroids early while preserving their endpoint-directed mean and their source. There is no separate force toward the global center; molecule centering fixes the translational degree of freedom.

### Mobile internal bridge

Let $`A(t)=[1-s(t)]^p`$. Write $`P_q`$ for subtraction of the mean within each fragment, and let

```math
F(t)=\begin{cases}
1,&t\le t_f,\\[2pt]
1-s\!\left(\dfrac{t-t_f}{1-t_f}\right),&t>t_f.
\end{cases}
```

The internal mean and scalar variance before projection are

```math
\mathbb E[q_t\mid q_0,q_1]=A(t)q_0+[1-A(t)]q_1,
\qquad
V_q(t)=\nu_qA(t)[1-A(t)]+\sigma_q^2[1-A(t)^2]F(t).
```

These are generated by

```math
dq_t=p\lambda(t)(q_1-q_t)\,dt+g_q(t)P_q\,dW_t^q,
\qquad g_q^2(t)=V_q'(t)+2p\lambda(t)V_q(t).
```

The mean rapidly approaches the local endpoint, but the mobility term maintains fluctuations of approximately $`\sigma_q`$ per local coordinate before projection. The fade beginning at $`t_f`$ removes these fluctuations at the endpoint. Internal structure can therefore remain mobile while fragments translate and rotate.

### Rotation bridge

Write $`[v]_\times w=v\times w`$ and
$`\Omega_f=\mathrm{Log}(R_{f,1}R_{f,t}^{\mathsf T})^\vee`$.
In the clock $`s`$, the rotation process has geodesic endpoint attraction and isotropic angular diffusion:

```math
dR_f=\left[\frac{\Omega_f}{1-s}\,ds
       +\frac{\rho}{\sqrt2}\circ dW_s^R\right]_\times R_f.
```

The circle denotes Stratonovich noise. Its action rotates all atoms of a fragment together about their centroid. The angular variance $`\rho^2=6`$ is one global constant shared by every fragment in both datasets; it does not depend on a reference template or enter the neural network as a feature.

## Fragment generator targets and loss

Set $`r_i=x_i-c_f=R_fq_i`$. The endpoint-dependent part of the pushed-forward coordinate drift is $`\lambda(t)B_i`$, where

```math
B_i=(c_{f,1}-c_f)+\Omega_f\times r_i
       +pR_f(q_{i,1}-q_i).
```

The conditional generator therefore has the form

```math
L_t^yf(X)=\lambda(t)\sum_i B_i\cdot\nabla_{x_i}f(X)
+\mathcal D_{t,x}f(X)+\mathcal Q_t^{a_1}f(X).
```

Here $`\mathcal D_{t,x}`$ is the known generator contribution from centroid, angular and internal noise, and $`\mathcal Q_t^{a_1}`$ is the label generator below. The driving noises are independent conditional on the endpoints; shared motions and centering introduce correlations between atoms. The rotation contribution includes its geometric drift correction. These known terms are retained by the sampler rather than learned as additional targets.

Translation, rotation and internal deformation can represent overlapping motions. We choose a common coordinate-based decomposition of $`B`$ rather than regressing the latent pieces independently. With fragment means denoted by a bar, define

```math
I_f=\frac1{n_f}\sum_{i\in I_f}
     (\|r_i\|^2I_3-r_ir_i^{\mathsf T}),\qquad
d_f^c=\overline B_f,
```

```math
d_f^R=I_f^{-1}\frac1{n_f}\sum_{i\in I_f}
       r_i\times(B_i-d_f^c),\qquad
d_i^q=\frac{B_i-d_f^c-d_f^R\times r_i}{p}.
```

The internal target has zero fragment mean and zero rotational component. This choice of **gauge** removes the ambiguity between shared motion and local deformation. It also explains the internal normalization: $`d^q`$ is paired with the faster rate $`p\lambda`$, giving

```math
\lambda(t)\bigl(d_f^c+d_f^R\times r_i+p\,d_i^q\bigr)
=\lambda(t)B_i.
```

The network predicts $`(\widehat d^c,\widehat d^R,\widehat d^q)`$ from the observed state and partition. Predictions use the same gauge, and centroid predictions are molecule-centered. The coordinate loss averages the following expression over fragments:

```math
\ell_f=
w_c(t)\|\widehat d_f^c-d_f^c\|^2
+w_R(t)(\widehat d_f^R-d_f^R)^{\mathsf T}I_f(\widehat d_f^R-d_f^R)
+\frac{w_q(t)}{n_f}\sum_{i\in I_f}\|\widehat d_i^q-d_i^q\|^2.
```

The inertia metric makes rotation error correspond to its displacement of the atoms. The positive component/time weights depend on observed quantities and time, so conditional regression still targets the corresponding conditional means. At generation time, the sampler recomputes centroids and residuals from the current coordinates and uses the learned fields; it does not need the training path's latent frames or endpoints.

## Atom-wise Brownian flow matching

The BM baseline uses the same source, labels and fragment conditioning, with a Cartesian bridge directly on $`x`$. There are no fragment translation, rotation or internal processes. Define a Kumaraswamy clock

```math
u(t)=1-(1-t^a)^b.
```

At a fixed time, its conditional marginal can be sampled as

```math
x_t=(1-u)x_0+ux_1+\sqrt{\beta u(1-u)}P\epsilon,
\qquad \epsilon\sim\mathcal N(0,I),
```

The time-coupled path is defined by the conditional SDE

```math
dx_t=\frac{u'(t)}{1-u(t)}(x_1-x_t)\,dt
       +\sqrt{\beta u'(t)}P\,dW_t.
```

One Cartesian head predicts $`d_i=x_{i,1}-x_{i,t}`$. Its conditional mean determines the marginal drift via the factor $`u'/(1-u)`$. The coordinate loss is

```math
\mathcal L_{\rm BM}
=\mathbb E\left[
\frac{w_x(t)}{(1.05-t)^2}\frac1{3n}\sum_i
\|\widehat d_i-(x_{i,1}-x_{i,t})\|^2\right].
```

The base loss weight is expressed in physical time $`t`$; it is not the square of the clock-dependent drift factor. It remains a positive weighting of endpoint regression.

The parameters $`a=2.3`$, $`b=3.1`$, $`\beta=14.8`$ were approximately fit to match the fragment conditional bridge's spread around its $`t=1`$ coordinates. For fixed centered endpoints, the BM mean-squared endpoint distance is

```math
\mathbb E\!\left[\frac1n\sum_i\|x_{i,t}-x_{i,1}\|^2\mid x_0,x_1\right]
=(1-u)^2\frac1n\sum_i\|x_{i,0}-x_{i,1}\|^2
+3\beta u(1-u)(1-1/n).
```

This matches an aggregate measure approximately; it does not give BM the fragment process's within-fragment correlations.

## Shared atom-label process and objective

For $`K`$ classes including the mask, define

```math
\kappa_1(t)=3t^2-2t^3,\qquad
\kappa_2(t)=0.2\kappa_1(t)[1-\kappa_1(t)],\qquad
\kappa_3(t)=1-\kappa_1(t)-\kappa_2(t).
```

The conditional label distribution is

```math
p_t(a_i\mid a_{i,1})
=\kappa_1\delta_{a_{i,1}}+\kappa_2\mathrm{Uniform}(K)
+\kappa_3\delta_{\rm mask}.
```

A label head predicts endpoint probabilities $`\pi_i^\theta`$. With $`\gamma=\kappa_3'/\kappa_3`$, the learned off-diagonal jump rate from current class $`a_i`$ to $`b\ne a_i`$ is

```math
q_t^\theta(b\mid a_i,X)
=(\kappa_1'-\gamma\kappa_1)\pi_i^\theta(b)
+\frac{\kappa_2'-\gamma\kappa_2}{K}.
```

Both models add the same label loss and mean-logit penalty to their coordinate objective:

```math
\mathcal L_{\rm label}
=\mathbb E\left[\frac1n\sum_i
\frac{w_a(t)}{3(1.2-t)}[-\log\pi_i^\theta(a_{i,1})]\right],
\qquad
\mathcal L_{\rm offset}
=0.01\,\mathbb E\left[\frac1n\sum_i
\left(\frac1K\sum_k\ell_{ik}\right)^2\right].
```

Here $`\ell_{ik}`$ are the logits. Their common additive offset leaves softmax probabilities unchanged; the penalty anchors that otherwise undetermined direction. The component/time weights are fixed within each training stage and calibrated as described in [Models and training](models-and-training.md#loss-balancing).
