# Lab4 Report
Lab4: Conditional Sequence-to-Sequence VAE  
Student: 林聯政 ID: 0856154  
Github link: https://github.com/summelon/DLP_Lab  

## Introduction
I implemented a conditional sequence-to-sequence VAE for English tense conversion and generation, include VAE model, dataset, training procudure and utilization for various usage and validation. I have tried various architecture modification or hyper-parameter tuning to make my model performs better and more robustness. These would be discussed later.
As a result, I obtained a score of `BLEU-4 score`: 85.10% and `Gaussian socre`: `0.4`. These are shown below.
![](https://i.imgur.com/CzXSRWX.png)

## Derivation of CVAE

## Implementation details
- Variational AutoEncoder(VAE) Architecture
    - Encoder
        - Word embedding(*, vocab_len -> *, hidden_size): Translate input words into embeddings
        - Condition embedding(4 -> 8): Translate 4 kinds of conditions into embeddings
        - Hidden_fc(8 -> hidden_size): Change last dimension of condition embedding to hidden size, reduce parameters
        - LSTM(*, hidden_size -> *, hidden_size):
            - Input shape: (seq_len, batch_size, hidden_size)
            - Use bidirection
            - Use multi-layer and dropout
        - Fc_mean, Fc_var(hidden_size -> rep_size): Translate lstm output to mean & log variance
        - Criterion(KL-Divergence): Use KL-Divergence to measure distance between the distribution
          of reparameterized latent and normal Gaussian distribution(mean=0, std=1)
    - Re-parameterization trick
        - If we sample z from a Gaussian distribution, we will face a problem that smapling operation doesn't have gradient.
          This problem is solved by the trick named `reparameteriztion` from original VAE paper.
          Reparameterization trick basically divert the non-differentiable operation out of the network, so that network can be trained.
          I follow the formula form the original paper: $x = \mu + \sum^{\frac{1}{2}}x_{std}$
    - Decoder
        - Word embedding(*, vocab_len -> *, hidden_size)   
          Although there are no input words in decoder, we still need word embedding in teacher forcing or iterative input the predictions
          from from last sequence in LSTM
        - Condition embedding(4 -> 8): Translate 4 kinds of conditions into embeddings
        - Hidden_fc(8+max_len(17)*output_size -> hidden_size)   
          The input of initial hidden contains a concatenation of latent and condition embedding.
          In order to match the input size of LSTM, here I add a linear layer to change their shape.
        - LSTM(*, hidden_size -> *, hidden_size):
            - Input shape: (seq_len, batch_size, hidden_size)
            - Use bidirection
            - Use multi-layer and dropout
        - Classifier(hidden_size -> vocab_size(29)): A Linear classifier change the LSTM output to model output
        - Criterion(Cross-Entropy)   
          Combine LogSoftmax() and NLLLoss in one. This loss function is used when training a classification problem with C classes.
    - Learn from variation
        - Input condition & latent space
            - In Encoder, I simply use the output of (condition -> embedding -> linear) as initial hidden input
            - In Decoder, Latent space from encoder or normal distribution and concatenate it with embedded condition will go through
              a linear, then as the input of initial hidden input in decoder
            - Note that under bidirectional or multi-layer LSTM, the first dimension of hidden will be num_direction * num_layers
    - Dataloader
        - I implemented a customized dataset class, it can:
            - Read data from .txt based on `pandas` and return (word, tense) pairs
            - change input data from character to index/one-hot or vise-versa
            - As an input feed into PyTorch dataloader class

- Specify the hyperparameters(KL weight, learning rate, teacher forcing ratio, epochs, etc.)
    - KL weight: I use cyclical schduler with period of 1e+4/batch_size
    - Learning rate: 5e-2
    - Batch size: 32
    - Teacher forcing ratio: 1 - KL ratio
    - epochs: 1000
    - Hidden size: 256
    - Number of LSTM layers in both encoder and decoder: 2
    - Dropout rate in LSTM: 0.5


## Results and discussion
- Show your results of tense conversion and generation and Plot the Crossentropy loss, KL loss and BLEU-4 score curves during training (5%)
    ![](https://i.imgur.com/ipSTWdJ.png)
- discuss the results according to your setting of teacher forcing ratio, KL weight, and learning rate.
    - Teacher forcing ratio  
      - Higher teacher forcing ratio will lead to better training result. While setting it to 1, the model will perform extremely bad
        on test score. The reason is that there won't be any absolute ground truth when validation. Then the model trained under teacher
        forcing is totally unexpected.
      - On the other hand, model training withou teacher forcing is going to be very hard. Since training a initialized model is difficult to
        generate a correct output to serve as next input. As a result, the model will learn nothing with low teacher focing ratio.
      - To solve the problem above, I use a cyclical teacher forcing ratio scheduler according to KL weight scheduler. In this way, model will
        be trained under various situation. In my opinion, this makes model more robustness.
    - KL-Divergence weight
      - Similar to teacher forcing ratio, it is not a good thing that KL weight is too large or too small all the time. KL weight should be small
        in the beginning, makes model adjust the updating smoother.
      - Here I implemented two scheduler method mentioned in the Lab4 introduction, which is `monotonic` and `cyclical` scheduler
      - From the figure we can find that cyclical scheduler converged faster than monotonic scheduler, which proves that it may perform better.
        The setting of both scheduler is mentioned in `Implementation details` part.
        ![](https://i.imgur.com/MVgmSdy.png)
    - learning rate
      - I set learning rate as 5e-2, which is the same as TA recommended. I have tried various value. While I found that smaller learning rate leads
        to slower convergence and larger learning rate result in unstable score. The training was keeping long. So I did not apply any learning rate
        scheduler due to the uncertainty of the coming better result.
