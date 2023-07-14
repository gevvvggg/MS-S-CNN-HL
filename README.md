# MS-S-CNN-HL
The DATASETS file includes the following datasets:

Felvoland dataset: C.mat, Alldata.mat, label.mat
San-Francisco dataset:C.mat, Alldata.mat, label.mat
Here are the corresponding Matlab codes for generating training data and labels, testing data, and predicting images with accuracy and calculating MAPE:

one_sample_1_1.m
main_data_all.m
main_data1206_label2
one_sample_main_data_W_200_Rw2_1.m
main_data_R.m
main_txt.m
mape.m
confusion2.m
These Matlab codes generate training data and labels for the Felvoland dataset:
1. one_sample_1_1.m - This code generates a single sample of training data and corresponding label for use in training a model.
2. main_data_all.m - This code generates all possible samples of training data and corresponding labels for use in training a model.
3. main_data1206_label2 - This code generates a subset of the Alldata.mat file from the Felvoland dataset, along with its corresponding labels, for use in training a model.
These Matlab codes generate test data for the Felvoland dataset:
4. one_sample_main_data_W_200_Rw2_1.m - This code generates a single sample of test data from the Alldata.mat file, along with its corresponding label, for use in evaluating the performance of a trained model.
5. main_data_R.m - This code generates all possible samples of test data from the Alldata.mat file, along with their corresponding labels, for use in evaluating the performance of a trained model.
These Matlab codes generate predictions for the Felvoland dataset and calculate accuracy, MAPE, and confusion matrix after testing all data:
6. main_txt.m - This code generates a text file containing the predicted labels for each test sample.
7. mape.m - This code calculates the Mean Absolute Percentage Error (MAPE) between the predicted labels and the true labels for each test sample.
8. confusion2.m - This code generates a confusion matrix showing the number of times each predicted label was assigned to each true label for each test sample.

The SUCON file contains both pre-training and training code.

1.The Fullgraphprediction.ipynb is used for graph prediction.
2.The dataenhancement.py script is used for data augmentation.
3.The losses.py script is used to define loss functions.
4."main_supcon.py is used for pre-training in contrastive learning, while main_ce and main_linear.py are used for training and fine-tuning. 
5.Util.py is used for some common utility functions and data processing methods. 
Specifically, it may include data preprocessing functions such as data cleaning, normalization, standardization, file operation functions like reading, writing, saving files, random number generation functions like generating random numbers and shuffling data order, statistical analysis functions like calculating mean, variance, standard deviation, and other auxiliary functions like string processing and time date processing. 
These functions are usually called by other Python scripts to implement data preprocessing, file reading and writing, random number generation, statistical analysis, and other functions in deep learning models."
