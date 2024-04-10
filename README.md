# Data Glitches Discovery using Influence-based Model Explanation

This repository contains the code of the paper Data Glitches Discovery using Influence-based Model Explanation.

## Installation and Running Instructions

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

The validation performances (accuracy on a validation set) of each model for a data glitch are reported in the table below. Note that the errors are presented in the training sets.

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


## Class-Based Detection for 5 runs with different random seeds
Comparison of F1-Score on 10% class-dependent noise detection between CNCI and existing influence signals. Note that in this case, 10% of the samples of each class are relabeled to another random class. CNCI is on par or better on detecting class-based mislabeled samples.

![mcb_sigs_cont_all](https://github.com/anonymoususr95/Influence-Based-Glitch-Detection/assets/159195769/fa0ce8b0-a9a7-44e3-a11e-c23d6c1b2f06)

## Unreduced F1-Score for Data/Model pairs for 5 runs with different random seeds

We report the unreduced performances from the comparative plots for the different models and datasets reported in the paper.

### Uniform Class Noise
![raw_mu_sigs](https://github.com/anonymoususr95/Influence-Based-Glitch-Detection/assets/159195769/63666b97-9230-409b-b926-09077075e1ec)

### Class-based Noise (contamination of one random class with 10% mislabeled samples)
![raw_mcb_sigs](https://github.com/anonymoususr95/Influence-Based-Glitch-Detection/assets/159195769/73b5e95e-ba1a-4406-b55c-f6d53d142c30)

### Anomalies
![raw_anom_sigs](https://github.com/anonymoususr95/Influence-Based-Glitch-Detection/assets/159195769/fc048f55-9a5e-40a2-855d-0f61ae3adb39)

## Additional Experiments 

### Experiments on Tabular Data 

In this experiment we used three deep learning models namely MLP, Resnet and FT-Transformer, that have been proven to be effective for various tabular datasets [NIPS '21](https://proceedings.neurips.cc/paper_files/paper/2021/file/9d86d83f925f2149e9edb0ac3b49229c-Paper.pdf). We employed three diverse datasets with different sample and feature size, namely Forest Cover, Jannis and Epsilon. We injected 10% uniform mislabeled samples in the training set of each dataset. Note that we followed the same evaluation pipeline described in our manuscript. In the figure below, we observe that the proposed CNCI signal outperforms all prior influence-based signals for the three models and datasets, detecting uniform mislabeled samples with 0.65 F1-score on average.

![additional_experiments](https://github.com/anonymoususr95/Influence-Based-Glitch-Detection/assets/159195769/ee7733bf-abfa-4b24-beed-97fd026251df)

### Ablation Study on Glitch Ratio

In this experiment we tried different anomaly and mislabeled ratios ranging from 1% up to 30%. 

Subsequently we present the influence-based signals F1-score for increasing mislabeled ratio. CNCI outperforms the influence-based signals in every dataset for both models, especially for low class-noise ratios. Specifically, for a 1% ratio CNCI achieves 36% better F1-score on average than SI (second-best signal).

![mislabelled_uniform_noise_exp](https://github.com/anonymoususr95/Influence-Based-Glitch-Detection/assets/159195769/82cec48e-4f45-4dec-a586-b21a7ca9ae46)

As depicted in the figure below, for the anomalous samples, the performance of PCID increases as the anomaly ratio increases. Note that the performance significantly decreases for smaller anomaly ratios. 

![ood_far_cl_noise_exp](https://github.com/anonymoususr95/Influence-Based-Glitch-Detection/assets/159195769/f199878d-7123-49ca-b425-df6f3399b877)

