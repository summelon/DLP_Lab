# Lab2 Report
Lab2: EEG Classification  
Student: 林聯政 ID: 0856154  
Github link: https://github.com/summelon/DLP_Lab  

## Introduction

## Experiment Setup
1. The detail of models
    - EEGNet  
    ![](https://i.imgur.com/kwBUbOz.png =50%x)
    - DeepConvNet  
    ![](https://i.imgur.com/DCLQklj.png =50%x)

2. Explain the activation function
    - ReLU  
    ![](https://i.imgur.com/s3OZPlF.png =50%x)
        - Pros
            - Computationally efficient—allows the network to converge very quickly
            - Non-linear—although it looks like a linear function, ReLU has a derivative function and allows for backpropagation
        - Cons
            - The Dying ReLU problem—when inputs approach zero, or are negative, the gradient of the function becomes zero, the network cannot perform backpropagation and cannot learn.
    - Leaky ReLU  
    ![](https://i.imgur.com/nFuUHXp.png =50%x)
        - Pros
            - Prevents dying ReLU problem—this variation of ReLU has a small positive slope in the negative area, so it does enable backpropagation, even for negative input values
            - Otherwise like ReLU
        - Cons
            - Results not consistent—leaky ReLU does not provide consistent predictions for negative input values.
        
    - ELU  
    ![](https://i.imgur.com/2ov2J6K.png =50%x)
        - Pros
            - It is continuous and differentiable at all points.
            - It is leads to faster training times as compared to other linear non-saturating activation functions such as ReLU and its variants.
            - Unlike ReLU, it does not suffer from the problem of dying neurons. This because of the fact that the gradient of ELU is non-zero for all negative values.
            - Being a non-saturating activation function, it does not suffer from the problems of exploding or vanishing gradients.
            - It achieves higher accuracy as compared to other activation functions such as ReLU and variants, Sigmoid, and Hyperbolic Tangent.
        - Cons
            - It is slower to compute in comparison to ReLU and its variants because of the non-linearity involved for the negative inputs. However, during the training times, this is more than compensated by the faster convergence of ELU. But during the test time, ELU will perform slower than ReLU and its variants.

## Experimental Results
1. The highest testing accuracy
    - Screenshot with two models
        - EEGNet  
        ![](https://i.imgur.com/7JGpYhF.png)
        - DeepConvNet  
        ![](https://i.imgur.com/OEl3ivw.png)

    - Anything you want to present
2. Comparison figures
    - EEGNet
    ![](https://i.imgur.com/XFsfyfU.png)
    - DeepConvNet
    ![](https://i.imgur.com/LCAt5UI.png)

## Discussion
1. Anything you want to share


## Reference
https://mlfromscratch.com/activation-functions-explained/#/
https://deeplearninguniversity.com/elu-as-an-activation-function-in-neural-networks/
https://missinglink.ai/guides/neural-network-concepts/7-types-neural-network-activation-functions-right/
