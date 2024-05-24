# Automated Waste Classification System

## Introduction

This project aims to address the urgent environmental and health challenges posed by current waste management practices by implementing an automated waste classification system. By utilizing advanced machine learning techniques, our system effectively differentiates waste types, promoting recycling, protecting nature, and fostering a sustainable future.

## Motivation and Impact

As global populations grow and consumerism surges, the volume of waste continues to increase, complicating waste management and classification processes. Our project seeks to disrupt the current situation by implementing an automated waste classification system that can significantly reduce the volume of waste consigned to landfills, thereby mitigating environmental degradation.

## Data Processing

The data was sourced from Kaggle ([Waste Classification Dataset](https://www.kaggle.com/datasets/techsash/waste-classification-data/data)), consisting of images depicting various types of waste such as paper, plastic, glass, and metal. The dataset was divided into training (80%) and test (20%) sets. Preprocessing steps included resizing images to 256x256 pixels, converting them to tensors, center cropping to 224x224 pixels, and normalization.

## Models Used

### 1. Multilayer Perceptron (MLP)
- **Architecture**: Three hidden layers with ReLU activations, followed by a softmax output layer.
- **Training**: Adam optimizer with a learning rate of 0.001, trained for 10 epochs with a batch size of 64.
- **Performance**: Test Accuracy: 84.28%, F1 Score: 0.841, Parameters: 38,577,226

### 2. Vision Transformer (ViT)
- **Model**: Pre-trained ViT_B_16
- **Training**: Adam optimizer with a learning rate of 1e-4, trained for 5 epochs.
- **Performance**: Test Accuracy: 90.45%, F1 Score: 0.881, Parameters: 85,806,346

### 3. Convolutional Neural Network (CNN)
- **Initial Architecture**: Three convolutional layers with increasing filter counts (32, 64, 128), followed by max-pooling layers, dropout, and batch normalization.
  - **Performance**:
    - CNN with Dropout: Test Accuracy: 89.38%, F1 Score: 0.863, Parameters: 51,605,826
    - CNN with Batch Normalization: Test Accuracy: 88.34%, F1 Score: 0.825, Parameters: 25,817,218
- **Enhanced Architecture**: Four convolutional layers (32, 64, 128, 256), batch normalization, ReLU activations, max pooling, and dropout.
  - **Performance**: Test Accuracy: 89.97%, F1 Score: 0.903, Parameters: 26,145,922

## Combined Results

| Model                          | Test Accuracy | # of Parameters | F1 Score |
|--------------------------------|----------------|-----------------|-----------|
| MLP                            | 84.28%         | 38,577,226      | 0.841     |
| ViT                            | 90.45%         | 85,806,346      | 0.881     |
| CNN with Dropout               | 89.38%         | 51,605,826      | 0.863     |
| CNN with Batch Normalization   | 88.34%         | 25,817,218      | 0.825     |
| CNN with Data Augmentation     | 87.78%         | 25,817,218      | 0.866     |
| Combined CNN                   | 89.97%         | 26,145,922      | 0.903     |

## Conclusion

Through rigorous experimentation and model evaluation, the Vision Transformer (ViT) and Combined CNN models emerged as promising candidates for waste classification tasks. While the ViT excelled in capturing global dependencies, its computational demands necessitate careful consideration. The Combined CNN model demonstrated reliable performance and practical utility.

Implementing these models can enhance waste management systems, improve the accuracy of sorting recyclables, reduce contamination, and increase recycling rates. Future work could include implementing k-fold cross-validation, expanding the dataset, and incorporating additional features to improve model accuracy and applicability.

## References

- [Waste Classification Dataset on Kaggle](https://www.kaggle.com/datasets/techsash/waste-classification-data/data)
