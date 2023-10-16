# Skin Lesion Classification
This repository contains the code for skin lesion classification on the [ISIC 2019 Skin Lesion images for classification](https://www.kaggle.com/datasets/salviohexia/isic-2019-skin-lesion-images-for-classification/data). 

## 1. Multiclass Classification 
First, a multiclass classification on the task in performed with 8 classes - 'MEL', 'SCC', 'VASC', 'AK', 'NV', 'BKL', 'DF', 'BCC'. The notebook can be found in `skin_lesion_mul.ipynb`. We do some initial data exploration to view the samples:
![alt text](http://url/to/img.png)
![alt text](http://url/to/img.png)
![alt text](http://url/to/img.png)

We use a pre-trained VGG16 architecture for the classification task. We finetune the model on the skin lesion data for 5 epochs. 

We get the following performance with this experiment:
![alt text](http://url/to/img.png)

Here's the confusion matrix:
![alt text](http://url/to/img.png)


## 2. Binary Classification
Second, we perform a binary classification task where 'SCC' and 'VASC' are treated as 'Non-tumour' (marked as NT in the notebook) category and the rest are 'Tumour' (marked as T in the notebook) category. Some initial data exploration shows that there is a huge imbalance in the data as shown below:
![alt text](http://url/to/img.png)

We do an initial experiment with a conventional cross-entropy loss and find that although the majority class performs very well(with 95% F1 score), the minority class only achieves 33% F1 score. A detailed performance is presented below:
![alt text](http://url/to/img.png)

Thus, to improve the performance we test 2 strategies here:
- Oversampling: Here we oversample the data using several techniques like:
    - Rotation(0, 10, 350, 355 degrees)
    - Vertical flipping
    - Horizontal flipping
    - Translation
    - Adding Gaussian noise 
    The class distribution improves as in:
    ![alt text](http://url/to/img.png)
    
    We find that training the model on the oversampled data does improve the performance of the model, especially boosting the performance on the minority class(`NT`).
- Class-weighted loss: Next, we try a weighted loss based on the distribution of samples between the 2 classes. In brief, since the minority class has lesser no. of samples, the model will be penalized more if it makes a wrong prediction on the minority class. Similarly, if it gives correct prediction on the minority class, it will be rewarded more than the majority class.