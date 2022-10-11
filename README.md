# Addressing leakage in Concept Bottleneck Models
This repository contains the code for the paper 'Addressing leakage in Concept Bottleneck Models' by M. Havasi, S. Parbhoo and F. Doshi-Velez


## Instructions

Install CUDA 11.1

Install the packages in the `requirements.txt`

Download the CUB-2011 dataset and replace the `CUB_200_2011` folder: https://www.vision.caltech.edu/datasets/cub_200_2011/ 

Run the data preprocessing script. This generates the labels and the concepts for each example.

```
python attribute_processing.py
```

Then run the training script:

```
python main.py
```

Notable parameters:
```
--save_metrics  # File to save the results in
--model_type # Hard or Soft CBM. Possible values: 'hard', 'independent', 'sequential', 'joint'
--lamb # Lambda parameter in soft joint CBM
--train_epochs # Number of training epochs
--latent_dims # Number of concepts in the side channel
--n_groups # Number of concept groups to use during training. This is used to model the scenario where only partial concepts are given. -1 means all concepts are used
--amortization_epochs # Number of epochs to train the amortization network
--pretrain_autoregressive # Number of epochs to train the autoregressive concept predictors while the body of the inception net is frozen
--label_multilayer # Use a two layer label predictor with this many hidden units
--autoregressive_multilayer # The autoregressive predictors have a hidden layer with this many units
--autoregressive # Whether to use an autoregressive concept predictor or not
--mc_samples_for_training # Number of MC samples during training
--mc_samples_for_prediction # Number of MC samples at prediction time
--inception_weight_decay # Weight decay
--batch_size # Batch size
--learning_rate # Learning rate
--dropout # Dropout rate
```
