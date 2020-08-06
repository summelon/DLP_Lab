# Lab3 Report
Lab3: Diabetic Retinopathy Detection  
Student 林聯政 ID: 0856154  
Github link: https://github.com/summelon/DLP_Lab

## Introduction  
In this lab, I implemented ResNet-V1 model, then trained it on the `Diabetic Retinopathy` dataset. It is an imbalance dataset with 5 classes, long tail distribution.
So there would be a big problem to solve the data imbalance. This is going to be discuss later.
I will also show the confusion matrix and transfer learning comparison figure to prove my exprimental results.   
![](https://i.imgur.com/ca80IzP.png)
## Experiment Setups
1. The details of ResNet
    - To implement ResNet-V1, I referred the official ResNet code in torchvision, while removed some redundant parts for other versions.
      I named my model the same name as torchvision does, to ensure successfully loading weights.
    - Main components in ResNet-V1
        - Be different from previous work, ResNet-V1 added a residual shortcut in each blocks, which prevents gradient vanishing and improves the convergence time.
        - Basic Block
            - Basict block consists of two 3X3 convolution followed by batch normalization adn relu respectively.
            - It is used in ResNet whose layers are less than 50
        - Bottleneck
            - Bottleneck is similar to separable convolution while the 3X3 convolution part is not a depthwise convolution.
            - Bottleneck is designed for reducing the parameters in deeper model. It separates a big normal convolution to a 1X1-3X3-1X1 convolution, while keeps the 
              same effect.
2. The details of your Dataloader    
    I implemented `open image`, `augmentation pipeline`, `weight sampler` and `image check` in my dataloader:
    - `open image`:  
        Simple image opening function based on Pillow
    - `augmentation pipeline`:  
        I tried two kinds of pipeline in this part, which includes `torchvision` and `imgaug`
        - `torchvision`:  
        There are several basic and useful transformations in torchvision library. In this part, I only implement `RandomHorizontalFilp`, `RandomVerticalFlip`,
        `RandomRotation` and `Normalization`.
        - `imgaug`:  
        It is a powerful github open source project for cv transformation. There are various kinds augmentation function with convenient condition in this library.
        In order to compare with `torchvision`, I added complex augmentations in this part, include different kinds of flip, affine, color changes and noise ...
    - `weight sampler`:  
        From EDA, I found that this is a imbalance dataset. To improve prediction more correct, it is necessary to add some balancing mechanism.
        `weight sampler` is based on the `RandomWeightedSampler` class in pytorch. It makes sample procedure in dataloader work like lottery. The possibility of each
        sample is referred to a given probabilistic list. Its length is same as index list.
    - `image check`:  
        This is a visualization function to check augmentation effect
3. Describing your evaluation through the confusion matrix  
    Confusion matrix shows the distribution(number) of predictions. Therefore we can intuitively know that how many image are misjudged to which other classes. Base on these
    observation, we have more evidence to decide how to adjust the imbalancing data. This section will be further discussion later.
    ![](https://i.imgur.com/GP2Qrvv.png)

## Experimental Results
1. The highest testing accuracy  
    ![](https://i.imgur.com/ySvEYsr.png)
2. Comparison figures
    - ResNet18:  
    ![](https://i.imgur.com/i4HAuiK.png)
    - ResNet50  
    ![](https://i.imgur.com/LfGZvdt.png)

## Discussion
1. Data balancing methods do not take advantage of this dataset:  
    - Due to the similar appearance among classes, model has trouble in classify classes whose distributions are very close.
      This is an imbalance dataset. Moreover, many augmentation can not be applied because of its limitation(symptom from details).
      For example, color jitter, Gussian noise or some other augmentation with pertubation against to the symptom takes minus effect.
    - Another solution is to balance the probability of occurence or the importance of each class, which implemented by `RandomWeightedSampler`
      and `WeightedLoss` respectively. However, according to the result from confusion matrix, I found that the accuracy of class 0 always drop after
      making other classes balance. Since class 0 is the most class, this kind of data balancing leads to worse result. Although others improved, we
      lost the most valuable class(highest percentage)
2. Higher image resolution works pretty good
    - As I mentioned in last item, data augmentation or smaple/loss balancing almost has no benifits from this dataset. Consequently another way is to
      make model more powerful.
    - Base on the observation from `EfficientNet`, Scaling up model entirely(width, depth, resolution) is helpful to improve the performance. Therefore I
      adjusted image resolution to 512X512 and scaled up model from ResNet18 to ResNet50, which makes model see and remember more details. The accruracy is
      improved from 79%(224X224, ResNet18) to 82%(512X512, ResNet50)

