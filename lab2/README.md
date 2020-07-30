# Lab2 Report
Lab2: EEG Classification  
Student: 林聯政 ID: 0856154  
Github link: https://github.com/summelon/DLP_Lab  

## Introduction
- In this lab, I implemented two kinds of neural network from scratch by PyTorch.
- In both networks, I compared the impact of different activation function according to their strategy and performance.
- For better test time accuracy, I have done some experiments to explore the relation of model performance and hyperparameter setting.

## Experiment Setup
1. The detail of models
    - EEGNet  
    ![](https://i.imgur.com/kwBUbOz.png =50%x)
        - EEGNet consists of a `firstConv` block, a `depthwiseConv` block, a `separableConv` block and `flatten` layer before a `linear` layer.
    - DeepConvNet  
    ![](https://i.imgur.com/DCLQklj.png =50%x)
        - There is a single convolution layer in DeepConvNet. It is followed by four basic block which consists of conv2d, bn, activation function, max pooling and a dropout layer. At last, be similar to EEGNet, there is a `flatten` layer followed by a `linear` layer.

2. Explain the activation function
    - ReLU  
    ![](https://i.imgur.com/s3OZPlF.png =50%x)
        - Pros
            - Easy for implement, low computational cost
            - Part of linear, but non-linearity
        - Cons
            - Neurons less than zero will die. The gradient of these parts become zero. These neuron won't be updated anymore.
    - Leaky ReLU  
    ![](https://i.imgur.com/nFuUHXp.png =50%x)
        - Pros
            - Small positive slope in the negative part, enable back propagation for negative input.
        - Cons
            - Inconsistent gradient descent may lead to unexpected results
        
    - ELU  
    ![](https://i.imgur.com/2ov2J6K.png =50%x)
        - Pros
            - It has not inconsistency problem compare to Leaky ReLU and dying neurons problem compare to ReLu.
            - Normally it should converge faster and achieve higher accuracy because of its merits.
        - Cons
            - It is worth using when training for faster convergence. On the other hand, it takes much more coputational cost in test time compare to ReLU and its variants.

## Experimental Results
1. The highest testing accuracy
    - Screenshot with two models
        - EEGNet  
        ![](https://i.imgur.com/7JGpYhF.png)
        - DeepConvNet  
        ![](https://i.imgur.com/OEl3ivw.png)
    - From the result, we can found that EEGNet is easier to train than DeepConvNet. In my opinion, because of the simple data in this time, shallow model convergences faster than complex model.
    - The experimental results also proves the advantage of activations I introduced above. The performance of ELU is always the best, which is as our expected. However, its shortcoming is hidden here, due to we do not measure the test time latency here.
2. Comparison figures
    - EEGNet
    ![](https://i.imgur.com/XFsfyfU.png)
    - DeepConvNet
    ![](https://i.imgur.com/LCAt5UI.png)

## Discussion
1. Different initialization will also affect the convergence speed and highest accuracy of the model.
2. In my experiments, use batch size as large as possible, takes advantage of better overall gradient updating. It means that we are able to update model to the most correct direction.
3. I found that suitable learning rate is important in this lab. If I use large learning rate, such as 1e-1 or 1e-2, later in the training phase, model loss may have fluctuation problem. On the other hand, too small learning rate will slow down the convergence time. Consequently, a learning rate in a suitable value is benifit for both rapid convergence time and smooth convergence curve

## Reference
https://mlfromscratch.com/activation-functions-explained/#/
https://deeplearninguniversity.com/elu-as-an-activation-function-in-neural-networks/
https://missinglink.ai/guides/neural-network-concepts/7-types-neural-network-activation-functions-right/
