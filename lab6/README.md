# Lab6 Report  
Lab6: Deep Q-Network and Deep Deterministic Policy Gradient  
Student: 林聯政 ID: 0856154  
Github link: https://github.com/summelon/DLP_Lab/tree/master/lab6

# Report
- A tensorboard plot shows episode rewards of at least 800 training episodes in:
    - __LunarLander-v2__  
    ![](https://i.imgur.com/snOpxnX.png)
    - __LunarLanderContinuous-v2__  
    ![](https://i.imgur.com/ex3N5FS.png)
- Describe your major implementation of both algorithms in detail
    1. Network architecture
        - LunarLander-v2:  
            - A 3 layers MLP for both bahavior network and target network
            - Input is 8 dimenstion(state)
            - Output is 4 discrete value(action)
        - LunarLanderContinuous-v2:
            - A 3 layers MLP for actor network, end with tanh() activation
            - Input is 8 dimenstion(state)
            - Output is 2 continuous value(action)
    2. Optimizer
        - Use Adam() for both
    3. Select action
        - Select action according to behavior model(no grad)
        - LunarLander-v2: epsilon-greedy  
            Use a factor `epsilon` to determine whether explore new action or not
        - LunarLanderContinuous-v2: Gaussian noise
            Add a noise sampled from Gaussian distribution to increase the exploration in continuous action space. Clip value if it is over the action space
    4. Update behavior network
        - Calculate Q value according to __behavior model__
        - Calculate expected Q value according to __target model__(no gradient)
            - If t + 1 is termination, Q value will be zero
            - Calculate $Y_i$ according to expectation Q value formula
    5. Update target network
        - Copy network weight from behavior models
    6. Test code in testing
        - Similar to training except model & memory buffer updating
- Describe differences between your implementation and algorithms
    - Loss function:  
        For better robustness, I use Huber loss instead of MSE loss. Huber loss acts like the mean squared error when the error is small, but like the mean absolute error when the error is large. This makes model update more stable when the estimates of Q are very noise.
    - Reward scaling when storing transition(0.1 in DQN, 0.01 in DDPG):  
        Some reward value may be perturbation, so we need add a scaling factor to reduce its effect.
    - Gradient clipping in DQN:  
        In the beginning, loss value might become very large since model has not learnt very good yet. We need apply gradient clipping to avoid loss value growing to NaN.
    - Action value clipping in DDPG:  
        The output of Actor model is after tanh(), therefore its value is in the range of (-1, 1). However adding a noise sampled from normal distribution, action value may be over (-1, 1). So we need to clip them if it is over the range.
- Describe your implementation and the gradient 
    - Actor updating:  
        1. Calculate present q value according to behavior critic network
        2. Calculate next action according to target actor network, then use next action and target critic network to find the next Q value
        3. Base on next Q value, calculate expection value: $Y = \gamma \times Q_{next} + R_{present}$. Next Q value will be zero if termination
        4. Loss function = F1_smooth_loss($Y_{expection}, Q_{present}$)
        5. Do backwardpropagation and update model
    - Critic updating
        1. Use behavior actor network to predict action according to present state
        2. To maximize the Q value, we have loss function: $\frac{1}{N}\sum_{i}Q_{behavior}(s, a | \theta^Q)$
        3. Do backwardpropagation and update model
- Explain effects of the discount factor  
    It describe the importance of multiple future reward to the current state.
- Explain benefits of epsilon-greedy in comparison to greedy action selection  
    Greedy algorithm may make the model fit in a sub-optimal policy. While we use epsilon-greedy, which contains a exploration probability ratio `epsilon`, to explore more possibility. Then use greedy algorithm to maximize our reward value.
- Explain the necessity of the target network  
    If we always update model according to behavior model, it is very unstable, especially for neuron network. So instead of using behavior model, we copy it after each C(100 in the code) steps as our update target, to ensure the stability of updating.
- Explain the effect of replay buffer size in case of too large or too small  
    If the replay buffer is too small, the data will be highly correlated. On the other hand, if it is too large, the data is likely to be outdated.

# Report Bonus
- Implement and experiment on Double-DQN
    - Implementation:  
        When update behavior model, instead of using target network to find the maximum of next Q value, we use behavior model to find next action. Then use this action and target model to find next Q value.
    - Training curve:  
    ![](https://i.imgur.com/tTjI7Px.png)

- Extra hyperparameter tuning  
    Not implement

# Performance
- LunarLander-v2: 274 ÷ 30 = 9
    - ![](https://i.imgur.com/AHX0Bxw.png)

- LunarLanderContinuous-v2: 240 ÷ 30 = 8
    - ![](https://i.imgur.com/8aAM2XM.png)
