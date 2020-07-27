# Lab 1 Report
Student: 林聯政 ID: 0856154   
- Github link: https://github.com/summelon/DLP_Lab  
## Introduction
From this lab, I have had a good command of the basic knowledge of simple neural networks.
I implemented one without any other frameworks but only numpy and some standard libraries.   
To further understand the impact of some components, 
I did comparison experiments of momentum and loss function, which is shown in `Discussion` part.
For convenient experiments, I wrote a argument parser as shown below:   
![](https://i.imgur.com/VUfV1gc.png =70%x)

## Experiment Setups
1. Sigmoid functions  
    - Original formula  
        $Sigmoid(\mathbf{X}) = \frac{1}{1+e^{-\mathbf{X}}}$
    - Derivative 
        $D(Sigmoid(\mathbf{x})) = \frac{e^{-\mathbf{x}}}{(1+e^{-\mathbf{x}})^2} = Sigmoid(\mathbf{X}) \circ (1-Sigmoid(\mathbf{X}))$ 
    - Implement: combined inside of `Linear` class
2. Neural networks
    - Overview graph  
        Input: (x, y) -> hidden layer1: [2, 8] -> hidden layer2: [8, 16] -> hidden layer3: [16, 1]  
        ![](https://i.imgur.com/mmaAw4o.png =50%x)
    - Linear layer
        - Original formula  
            $Linear(\mathbf{X}) = \mathbf{W} \cdot \mathbf{X} + \mathbf{B}$
        - Derivative for weight  
            $D(Linear(\mathbf{X})) = \mathbf{X}$
3. Backpropagation  
    - Loss function
        - Cross Entropy
            - Formula   
                $CE(y, \hat{y}) = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{C} \hat{y}^{(i)}_j \ln{y^{(i)}_j}$
            - Derivative  
                $D(CE(y, \hat{y})) = -\frac{1}{N} (-\frac{\hat{y}^{(i)}}{y^{(i)}} + \frac{1-\hat{y}^{(i)}}{1-y^{(i)}})$
        - Mean Squared Error
            - Formula  
                $MSE(y, \hat{y}) = \frac{1}{NC} \sum_{i=1}^{N} \sum_{j=1}^{C} (y_j^{(i)} - \hat{y}_j^{(i)}) ^ 2$
            - Derivative  
                $D(MSE(y, \hat{y})) = \frac{2}{N} (y_j^{(i)} - \hat{y}_j^{(i)})$
    - Momentum
        - Formula   
           $\mathbf{W} = \mathbf{W} - (lr * \mathbf{G_W} + gamma * \mathbf{V_W})$ 
    - Backward example
        - Weight in linear3:    
            $G_{W3} = \frac{\partial Loss}{\partial Act} \cdot \frac{\partial Act}{\partial Linear} \cdot \frac{\partial Linear}{\partial W_3}$
        - Derivative detail of each parts have been shown above

## Results of Testing
- Test setting:
    - Loss function: Mean Squared Error(MSE) 
    - learning rate: 1e-1
    - momentum(gamma): 9e-1
1. Screenshot and comparison figure  
    - Linear data:   
        ![](https://i.imgur.com/ME9rLFe.png =50%x)
    - Xor data:  
        ![](https://i.imgur.com/uup6VpN.png =50%x)
2. Show the accuracy of your prediction
    - Linear data:   
        ![](https://i.imgur.com/s5EQgXx.png =10%x)
        ![](https://i.imgur.com/RI3lYkP.png =10%x)
        ![](https://i.imgur.com/lFBTWfV.png =10%x)
        ![](https://i.imgur.com/TM78FX7.png =60%x)
    - Xor data:   
        ![](https://i.imgur.com/Weu6C8Z.png =15%x)
        ![](https://i.imgur.com/sLQRmTp.png =60%x)
3. Learning curve(loss, epoch curve) 
    - Linear data:  
        ![](https://i.imgur.com/XJi5j6m.png =50%x)
    - Xor data:   
        ![](https://i.imgur.com/aYiyqNj.png =50%x)

## Discussion
1. Try different learning rates
    - Test data: Xor
    - Almost the same setting as `Results of Testing`, only change learning rate from `1e-1` to `1e-3`   
      the __Left side__ shows the learning curve of `1e-1` and the __right side__ shows the learning curve of `1e-3`
      ![](https://i.imgur.com/aYiyqNj.png =40%x)
      ![](https://i.imgur.com/QqtPAru.png =40%x)
    - Found
      Two figure are nearly the same except two points:
      1. Training model with `1e-1` learning rate(left) is faster(less epoch)
      2. Training model with `1e-3` learning rate(right) is smoother(see the learning curve at 0-50 epoch)   
      - As mentioned in the introduction from TAs, the figures confirm that the greater the ratio, 
        the faster the neuron trains; the lower the ratio, the more accurate the training is.
2. Try different numbers of hidden units
    - Test data: Linear, change `n` to 10000
    - Hyperparameter setting is exactly the same as `Result of Testing`
    - | Hidden Layers    | \[8, 16](Left) | \[32, 64](Middle) | \[128, 256](Right) |
      |:---------------- | -------------  | ----------------  |:-----------------:|
      | Epochs(100% acc) | 500            | 400               |        100        |
      | Training Time    | 1m9s           | 1m33s             |       1m46s       |
      ![](https://i.imgur.com/hribi8U.png =30%x)
      ![](https://i.imgur.com/3MYYltZ.png =30%x)
      ![](https://i.imgur.com/83ZglLC.png =30%x)
    - Found
        1. More hidden units are more powerful and faster to fit the data
        2. Less hidden units need less computational resouces and may take less time under simple task
        3. More hidden units might need lower learning rate.
           I uesed learning rate as `1e-1` here.
           The more hidden units, the serious fluctuation problem.
           Therefore use lower learning rate in more hidden units may be trained smoother.
3. Try without sigmoid function
    - No matter in `linear` data or `xor` data,
      when I trained the model without sigmoid function, the loss would always become __larger and larger__.
      From this view, we may consider sigmoid function as not only a activation function, but also a clip function.
      It prevents the output of the model over the range of [0, 1].
      As a result, the loss between ground truth(0 or 1) and outputs won't be too larger.
4. Momentum
    - Use exactly the same setting as `Result of Testing` to show the effeciency of momentum
    - Comparison:   
        with momentum: converge in 200 epochs(left)    
        without momentum: converge in 1800 epochs(right)    
        ![](https://i.imgur.com/XJi5j6m.png =40%x)
        ![](https://i.imgur.com/dWPIBpE.png =40%x)
    - Found   
        It is more efficiency if we use momentum to update the model.
        Although it may take some unexpected fluctuation.
5. Nesterov Momentum   
    I implemented nesterov momentum in this lab.
    While I found that no matter I use it or not, the model will always converged in the same epoch under the same setting.
    I think the reason is that these classification tasks are too simple.
    So the Nesterov Momentum cannot show its power.
6. Loss function
    - Compare Mean Squared Error(MSE, left) to Cross Entropy(CE, right)   
        ![](https://i.imgur.com/ak4dCTY.png =40%x)
        ![](https://i.imgur.com/7sugBQH.png =40%x)
    - Found
        1. Although normally we think CE(Binary Cross Entropy in this case actually) is better than MSE,   
           in this lab it is not that case.
           The figure depicts that model update through MSE is faster(less epoch)
        2. From another point, BCE is more stable compare to MSE(less fluctuation)

## Reference
- [Link](https://mlfromscratch.com/neural-network-tutorial/#/)
  Neural Network from Scratch: black page, introduce how to backward
- [Link](https://math.stackexchange.com/questions/945871/derivative-of-softmax-loss-function)
  Derivative of Softmax loss: derive formula
- [Link](https://zhuanlan.zhihu.com/p/58964140)
  Neural network realization with example(zhihu)
- [Link](https://medium.com/jarvis-toward-intelligence/%E6%AF%94%E8%BC%83-cross-entropy-%E8%88%87-mean-squared-error-8bebc0255f5)
  Compare CE, MSE, Sigmoid, Softmax
