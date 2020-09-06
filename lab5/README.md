# Lab5 Report  
Lab5: Let's play GANs  
Student: 林聯政 ID: 0856154  
Github link: https://github.com/summelon/DLP_Lab/tree/master/lab5

## Introduction  
- In this lab, I implemented a multi-label conditional GAN model. In order to improve the test classification accuracy, I have tried 3 kinds of GAN variations, including ACGAN, WGAN and WGAN-GP, they are all based on a DCGAN architecture and a little different in details.
- As a result, my test classification accuracy is 79%

## Implementation details
- Describe how your implement your model
    - Choice of cGAN
        - Auxiliary Classifier GAN(ACGAN)
        - Wasserstein GAN(WGAN)
        - Wasserstein GAN Gradient Penalty(WGAN-GP)
    - Model architectures
        - I refered the DCGAN achitecture from PyTorch tutorial
        - Generator:
            - Consist of multiple block of ConvTranspose-(BN)-ReLU, not use BN in WGAN
            - Tanh() activation in last layer: to ensure the model treat bright and dark pixel fairly
        - Discriminator
            - Consist of multiple bock of Conv-(BN)-LeakyReLU, not use BN in WGAN
            - Sigmoid activation in last layer: to discriminate real or fake, not use in WGAN
    - Loss functions
        - Normal: Binary Cross-entropy(BCE) for single class classification(real or fake)
        - Auxiliary: BCE for multi-label classification
        - WGAN: Wasserstein distance/Earth-moving distance to calculate the distance between two distribution
- Specify the hyperparameters
    - `learning rate`: 2e-4
    - `epoch`: 1000
    - `batch size`: 64
    - `Optimizer`: Adam with b1=0.5, b2=0.999
    - `latent dim`: 100
    - `image size`: 64x64

## Results and discussion
- Show your results based on the testing data
    - Generated images:  
    ![](https://i.imgur.com/wA55pTS.png)
    - Test result: 79% accuracy   
    ![](https://i.imgur.com/l8T9463.png)
- Discuss the results of different models architectures
    - Noise
        In order to make discriminator learn more robustness, a random(0.5 ratio) generated wrong labels are used with real image(the other pair is generated image with respective labels) as input for discriminator. This makes discriminator focus on the image details instead of figure out the different distribution of real image and fake image
    ![](https://i.imgur.com/6cdVJGH.png)
    - Concatenate condition(ACGAN):  
        Different with conventional CGAN or InfoGAN, I used ACGAN, which has an additional classification loss for better condition learing. The way condition and latent space concatenate is shown above. While the performance of ACGAN was not good as my expectation, which is nearly 60% test classification accuracy.
    - Loss function(WGAN):
        To further improve my model, I kept the model architecture but change the loss function into Wasserstein distance. In the beginning, target distribution is too far to approximate by the initialized model distribution, since there are a few intersections between them. While Wasserstein distance is a good measurement in this situation, it is much more efficient than other loss function to help model to update.  
        While due to the limitation of WGAN, its best performance is even worse than ACGAN, which is 54% test classification accuracy.
    ![](https://i.imgur.com/4IotTEj.png)
    - Improve loss function(WGAN-GP):
        There is a shortcoming in original WGAN, which is the limitation of Lipschitz continuous assumption. Due to this, we need to prevent gradient explosion by gradient clipping. This method works, while make gradient accumulate on the upper bound and lower bound, just like the figure above. To improve this, I applied gradient penalty in WGAN, which makes gradient distribution as a normal distribution. It brougnt further improvement in test classification accuracy, which is 79%.
