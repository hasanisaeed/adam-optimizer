# Adam Optimizer
Implementing the Adam optimizer to understand the mathematics behind it.


#### Initialize:
- Initialize the first moment vector:
```math
m_0 = 0
```
- Initialize the second moment vector:
```math
v_0 = 0
```
- Initialize the timestep:
```math
t = 0
```
#### Update Rule:
1. Update the timestep:
```math
t = t + 1
```
2. Compute the gradient:
```math
g_t = \nabla_\theta f_t(\theta_{t-1}
```
3. Update biased first moment estimate:
```math
 m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t
```
4. Update biased second raw moment estimate: 
```math
v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2
```
5. Compute bias-corrected first moment estimate:
```math
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}
```
6. Compute bias-corrected second raw moment estimate:
```math
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
```
7. Update the parameters:
```math
\theta_{t+1} = \theta_t - \frac{\alpha \cdot \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
```
 

 
