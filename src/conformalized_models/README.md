# Conformal Predictors

This folder contains the implemented conformal predictors

The main idea is that these models are implemented in a class style, in which they provide two main methods.

The first method, `fit()`, is responsible for using the data from the the calibration data collection to initialize the method.
For example, in the Vanilla case, it calculates the $1-\alpha$ quantile of the non-conformity scores.

The second method `predict()`, is responsible for generating intervals of uncertainty around the point forecast values for the
next `config.HORIZON` values (i.e., $\hat C(\mathbf{x}\_{t+k|t}) = [\hat \rho_i(\mathbf{x}\_{t+k|t}) \pm \hat q]$). This means that, at time
$t$, given forecasted states of the grid for timesteps $(t+1, \ldots, t+12)$, the predict method provides intervals of uncertainty for each
horizon, given the point predictions of $\hat \rho(x_{t+k|t})$, for $k \in (1, \ldots, 12)$.

## `vanilla.py`

The Vanilla (split conformal prediction) method constructs prediction intervals using only the residuals of the calibration set.

Let $\mathcal{T}_{\text{cal}}$ denote the calibration set.

For each transmission line $i$, the absolute residuals are used as nonconformity scores, and a global quantile is computed as:

$$\hat{q}_{i,1-\alpha} = quantile (\underbrace{\cup_{t_s\in \mathcal T_{cal}} \{|\rho_{i,t_s} - \hat{\rho_i}(\mathbf{x}_{t_s})|\}}_\text{abs residuals in calibration set}, 1-\alpha )$$

At test time, this quantile is used to construct a prediction interval around the point forecast $\hat \rho_i(\mathbf{x}_{t+k})$:

$$\hat{C}_{i, 1-\alpha}(\mathbf{x}_{t+k}) = [\hat{\rho}_i(\mathbf{x}_{t+k}) - \hat{q}_{i,1-\alpha}; \hat{\rho}_i(\mathbf{x}_{t+k}) + \hat{q}_{i,1-\alpha}]$$

## `knn_norm.py`

This variant uses uncertainty scalars to scales nonconformity scores using a local difficulty estimate.
The difficulty estimate, calculated for each $\mathbf{x}_t$ in the calibration dataset, averages the residuals of the $k$ most similar neighbours of the calibration dataset:

$$\text{difficulty}(\mathbf{x}_t) = \max(\text{mean} ( \underbrace{\cup_{t_s\in \mathcal{N}_k(\mathbf{x}_t | \mathcal T_{cal})} |\rho_{i,t_s} - \hat{\rho}_{i}(\mathbf{x}_{t_s})|}_{\substack{\text{set of $k$ closest neighbours} \\ \text{in calibration set}}}), \epsilon)$$

where $\varepsilon > 0$ prevents numerical instability.

Each calibration residual is normalised as:

$$s_{i}(\mathbf{x}_t) = \frac{|\rho_{i, t} - \hat{\rho_i}(\mathbf{x}_t)|}{\text{difficulty}_i(\mathbf{x}_t)}$$

and a global quantile is then computed from the normalized scores:

$$\hat{q}_{i,1-\alpha}^{(\text{norm})} = \text{quantile} \left(\underbrace{\cup_{t\in \mathcal T_{cal}} \{ s_{i}(\mathbf{x}_t) \}}_{\text{normalised residuals}}, 1-\alpha \right)$$

At test time, given a new forecasted system state $\mathbf{x}_{t+k}$, the prediction interval is obtained by re-scaling this quantile:

$$\hat{C}_{i, 1-\alpha}(\mathbf{x}_{t+k}) = \left[\hat{\rho}_{i}(\mathbf{x}_{t+k}) - \hat{q}_{i, 1 - \alpha}^{(norm)}\times \text{difficulty}_i(\mathbf{x}_{t+k}); \hat{\rho}_{i}(\mathbf{x}_{t+k}) + \hat{q}_{i,1-\alpha/2}^{(norm)}\times \text{difficulty}_i(\mathbf{x}_{t+k})\right]$$

## `aci.py`

The ACI method dynamically updates the significance level $\alpha^\ast$ in an online fashion, using a gradient descent-like procedure in which:

- if intervals are too narrow (miss too often) $\alpha^\ast$ decreases, producing wider intervals
- if intervals are too wide (coverage above target) then $\alpha^\ast$ increases, shrinking intervals

The adaptive significance level is updated by following the ACI update rule:

$$\alpha^\ast_{i, t} = \alpha^\ast_{i, t-1} + \gamma\left(\alpha - \mathbf{1} \left( \rho_{i, t-1} \notin \hat{C}_{i, 1-\alpha^\ast_{i, t-1}}(\hat \rho_i(\mathbf{x}_{t-1})) \right)\right)$$

where $\gamma$ is the step size parameter. The ACI method starts with $\alpha^\ast=\alpha$.

The quantile is calculated for all forecasted states $(\mathbf{x}\_t, \ldots, \mathbf{x}\_{t+k})$ using the adaptive significance level at time $t$, corresponding to $\alpha_{i, t}^{\ast}$, and is updated sequentially in a delayed fashion when the real simulation reaches time $t+k$.

$$\hat{q}_{i, 1-\alpha^\ast_{i, t}} = \text{quantile} \left(\underbrace{\cup_{t_s\in \mathcal T_{cal}} \{|\rho_{i,t_s} - \hat{\rho}_i(\mathbf{x}_{t_s})|\}}_{\text{abs residuals in calibration set}}, 1-\alpha^\ast_{i, t} \right)$$

The prediction interval is then:

$$\hat{C}_{i, 1-\alpha_{i,t}^\ast}(\mathbf{x}_{t+k}) = \left[\hat{\rho}_{i}(\mathbf{x}_{t+k}) - \hat{q}_{i,1-\alpha^\ast_{i, t}}; \hat{\rho}_{i}(\mathbf{x}_{t+k}) + \hat{q}_{i,1-\alpha^\ast_{i, t}}\right]$$

This adaptive mechanism enables asymptotic coverage guarantees under distribution shift, making ACI particularly interesting for evolving power system operations.

## `knn_norm_with_aci.py`

Experimental combining the uncertainty scalars from the `knn_norm.py` and the adaptive alphas/quantiles from `aci.py`
