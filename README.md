# adam-optimizer
> This is an implementation in C language for a better understanding of the mechanics behind the Adam Optimizer, based on the article by [Cristian Leo](https://towardsdatascience.com/the-math-behind-adam-optimizer-c41407efe59b).

#### Initialize:
- Initialize the first moment vector: $m_0 = 0$
- Initialize the second moment vector: $v_0 = 0$
- Initialize the timestep: $t = 0$

#### Update Rule:
1. Update the timestep: $$t = t + 1$$
2. Compute the gradient $g_t$: $$g_t = \nabla_\theta f_t(\theta_{t-1})$$
3. Update biased first moment estimate: $$m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t$$
4. Update biased second raw moment estimate: $$v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2$$
5. Compute bias-corrected first moment estimate: $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$
6. Compute bias-corrected second raw moment estimate: $$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
7. Update the parameters: $$\theta_{t+1} = \theta_t - \frac{\alpha \cdot \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$
