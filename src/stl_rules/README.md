# STL background

Let $\rho_t$ denote the capacity of a certain line at time $t$. The requirement for the line to be safe
can be expressed with the following STL formula:

$$\phi := G_{[1, 12](\rho \leq 0.95)}$$

which is tru iff all true $\rho$ values are below 0.95.

Considering the function:

$$h(\rho) = 0.95 - \rho$$

then, by the semantics of the operator, the robustness of $\phi$ at time $t$ is expressed as:

$$\psi^\phi(\rho, t)=\min _{k=1, \ldots, 12} h\left(\rho_{t+k}\right)=\min _{k=1, \ldots, 12}\left(0.95-\rho_{t+k}\right)$$

If $\psi^\phi(\rho, t) > 0$, then the specification holds, i.e., the line is secure for the next 12 timesteps.

## Direct STL Predictive Runtime Verification

Based on paper "Conformal prediction for stl runtime verification"

Suppose we have point forecasts $\hat \rho_{t+1}, \ldots, \hat \rho_{t+12}$, generated at time $t$.
The predicted robustness is:

$$\psi^\phi(\hat \rho, t) = \min _{k=1, \ldots, 12}\left(0.95-\hat\rho_{t+k}\right)$$

For each trajectory $i$ (typically oone trajectory of 12 values an hour) in the calibration set, we compute the non-conformity score:

$$R^{(i)}=\psi^\phi(\hat{\rho}^{(i)}, t)-\psi^\phi(\rho^{(i)}, t)$$

Let $C$ be the ($1-\alpha$)-quantile of { $R^{(i)}$ }.
The idea is that this value of C is constructed to satisfy:

$$P\left(\psi^\phi(\hat{\rho}, t)-\psi^\phi(\rho, t) \leq C\right) \geq 1-\alpha$$

and if $\psi^\phi(\rho, t) > C$, then $P((\rho, t) \models \phi) \geq 1-\alpha$.
This would mean the trajectory is safe with at least $(1-\alpha)$ probability.

### `vanilla_rule.py`

This method implements the Direct approach rule from above.

### `knn_norm_rule.py`

Similar strategy to the `conformalized_models/knn_norm.py`.

The same rule is implemented, but instead of having a single global threshold `C` calculated with the non-conformity scores, we first normalise the non-conformity scores, and then calculate a normalized `C_norm` by using the difficulty metric.

The difficulty estimate, calculated for each trajectory $T$ in the calibration dataset, averages the residuals of the $k$ most similar **trajectory** neighbours of the calibration dataset:

$$\text{difficulty}(T) = \max(\text{mean}(\underbrace{\cup_{t\in \mathcal{N}_k(T | \mathcal T_{cal})} (\psi^\phi(\hat{\rho}{}^{(i)}, t)-\psi^\phi(\rho^{(i)}, t))}_{\text{set of k nearest neighbours in calibration set}}), \epsilon)$$

where $\varepsilon > 0$ prevents numerical instability.

For each trajectory T, its calibration residual is normalised as:

$$S_{i}^{(norm)}(T) = \frac{|\psi^\phi(\hat{\rho}^{(i)}, T)-\psi^\phi(\rho^{(i)}, T)|}{\text{difficulty}_i(T)}$$

and a global quantile is then computed from the normalised scores:

$$\hat{q}_{i,1-\alpha}^{(\text{norm})} = \text{quantile} (\underbrace{\cup_{t\in \mathcal T_{cal}} \{ S_{i}^{(norm)}(t) \}}_{\substack{\text{normalised trajectory} \\\ \text{residuals}}}, 1-\alpha)$$

At test time, given a new forecasted trajectory $T_{new}$, we calculate its difficulty, and verify:

If $\psi^\phi(\rho, t) > C_{norm} \times \text{difficulty}(T_{new})$, then trajectory is safe. Otherwise, unsafe.
