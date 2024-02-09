# Data Glitches Discovery using Influence-based Model Explanation

Here is the code of the paper Data Glitches Discovery using Influence-based Model Explanation.

## Training Settings 
Subsequently we report per dataset the learning rate, batch size and epochs that we train each foundational model. 

| Dataset       | Model    | Learning Rate | Batch Size | Epochs |
|---------------|----------|---------------|------------|--------|
|     MNIST     | ResNet   |          0.01 |        128 |      7 |
|     MNIST     | ViT      |           0.1 |        128 |      2 |
|     MNIST     | ConvNeXt |          0.05 |        128 |      2 |
| Fashion MNIST | ResNet   |          0.01 |        128 |      7 |
| Fashion MNIST | ViT      |           0.1 |        128 |      2 |
| Fashion MNIST | ConvNeXt |           0.1 |        128 |      4 |
|    CIFAR-10   | ResNet   |          0.01 |        128 |      7 |
|    CIFAR-10   | ViT      |           0.1 |        128 |      4 |
|    CIFAR-10   | ConvNeXt |           0.1 |        128 |      3 |

## Validation Performance 

The validation performance of each model for a data glitch are reported in the table below. Note that the errors are presented in the training sets.

| Dataset       | Model    | Uniform Class Noise | Class-based Noise | Anomalies       |
|---------------|----------|---------------------|-------------------|-----------------|
|     MNIST     | ResNet   | 0.94 $\pm$ 0.01     | 0.95 $\pm$ 0.01   | 0.96 $\pm$ 0.01 |
|     MNIST     | ViT      | 0.83 $\pm$ 0.03     | 0.81 $\pm$ 0.03   | 0.90 $\pm$ 0.01 |
|     MNIST     | ConvNeXt | 0.83 $\pm$ 0.01     | 0.82 $\pm$ 0.02   | 0.87 $\pm$ 0.01 |
| Fashion MNIST | ResNet   | 0.82 $\pm$ 0.01     | 0.80 $\pm$ 0.01   | 0.82 $\pm$ 0.01 |
| Fashion MNIST | ViT      | 0.81 $\pm$ 0.01     | 0.81 $\pm$ 0.01   | 0.83 $\pm$ 0.01 |
| Fashion MNIST | ConvNeXt | 0.81 $\pm$ 0.01     | 0.80 $\pm$ 0.01   | 0.82 $\pm$ 0.01 |
|    CIFAR-10   | ResNet   | 0.89 $\pm$ 0.01     | 0.88 $\pm$ 0.00   | 0.90 $\pm$ 0.00 |
|    CIFAR-10   | ViT      | 0.90 $\pm$ 0.01     | 0.90 $\pm$ 0.01   | 0.91 $\pm$ 0.01 |
|    CIFAR-10   | ConvNeXt | 0.85 $\pm$ 0.01     | 0.85 $\pm$ 0.00   | 0.87 $\pm$ 0.00 |


## Class-Based Detection
F1-Score on 10% class-noise detection of CNCI and existing influence signals. Note that the 10% of the samples of each class is labeled to another random class. CNCI is on par or better on detecting class-based mislabeled samples.

![mcb_sigs_cont_all](https://github.com/anonymoususr95/Influence-Based-Glitch-Detection/assets/159195769/fa0ce8b0-a9a7-44e3-a11e-c23d6c1b2f06)

