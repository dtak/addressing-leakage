# Addressing leakage in Concept Bottleneck Models
This repository contains the code for the paper 'Addressing leakage in Concept Bottleneck Models' by M. Havasi, S. Parbhoo and F. Doshi-Velez


## Setup instructions

Install CUDA 11.1

Install the packages in the `requirements.txt`

Download the CUB-2011 dataset and replace the `CUB_200_2011` folder: https://www.vision.caltech.edu/datasets/cub_200_2011/ 

Run the data preprocessing script. This generates the labels and the concepts for each example.

```python attribute_processing.py```
