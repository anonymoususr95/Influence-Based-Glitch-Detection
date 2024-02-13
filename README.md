# Data Glitches Discovery using Influence-based Model Explanation

This tepository contains the code of the paper Data Glitches Discovery using Influence-based Model Explanation.

## Installation and Running Instructions

### Experimental Setting

**Note**: The experiments run on a 16-core Intel Xeon cpu, 64GB ram and no gpu. In our experiments we have used Python 3.9.

Install the necessary packages by running `pip install -r requirements.txt`

Run `make demo_single` that runs CNCI (for uniform-class noise, class-dependent noise)  and PCID (anomalies) for ResNet-20 on the Fashion MNIST dataset. The output is a barplot showing the F1-score of our signals CNCI (or PCID) w.r.t. the existing influence-based signals (SI, MAI, MI and GD-class).

Run `make demo_mixed` that runs CFRank (the proposed mixed signal), CNCI, and PCID for ResNet-20 on the MNIST dataset with mixed errors, both mislabeled and anomalous samples. The output (printed in the console) is the F1-scores of the three signals and the error characterization accuracy, i.e., how accurately a detected error is being characterized.

**Note**: In case of memory error, consider *decreasing* the "batch_size" in the json file "configs/resnet/tracin_resnet.json". 

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
Comparison of F1-Score on 10% class-dependent noise detection between CNCI and existing influence signals. Note that in this case, 10% of the samples of each class are relabeled to another random class. CNCI is on par or better on detecting class-based mislabeled samples.

![mcb_sigs_cont_all](https://github.com/anonymoususr95/Influence-Based-Glitch-Detection/assets/159195769/fa0ce8b0-a9a7-44e3-a11e-c23d6c1b2f06)

