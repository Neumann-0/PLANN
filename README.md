# PLANN
## Introduction
In recent years, PLNN models have received widespread research and application because of their simplicity and ease of interpretation, and much progress has been made in the interpretation of PLNNs, which has theoretically demonstrated their equivalence and provided a basis for studying the interpretation of more complex models.In this paper, we propose a new super-linear region-based interpretation of the instance prediction results of PLNN models.
## How to run the code
1. Use "SDIP.py" to train a PLNN model by employing the data in the training set . The trained model will be saved in "*.pt".  
2. Use "MnistShapelets" to interpret the picture, you should put a picture and a trained PLNN model as input, then you will recieve a picture with some parts of the original picture as the output, that is the interpretation.
## Usage
1. train the PLNN model.  
$ python SDIP.py --stage <train|discover_shapelet> -train_data_path  <traindata> -test_data_path <testdata> -model_path <modelfile> -datasize <datasize> -label_format <[0|1]> [-H1 \<H1>\] \[-H2 \<H2>\] \[-H3 \<H3>\] \[-epochs \<epochs>\]  
2. interpret a picture  
Input: a picture and a trained PLNN model  
Output: an interpreted picture
## Parameters
-stage train : train a PLNN model  
  
-train_data_path : set the filename of training dataset  
  
-test_data_path : set the filename of test dataset  
  
-model_path : set the modelfile name for saving model  
  
-datasize : set the lengh of an instance time series of training dataset  
  
-label_format : set the label format, 0 or 1  
                 &emsp;0: the class label of time series dataset is start from 0  
                 &emsp;1: the class label of time series dataset is start from 1  
  
-H1 : set the number of neurons for the first hidden layer of PLNN model  
  
-H2 : set the number of neurons  for the second hidden layer of PLNN model  
  
-H3 : set the number of neurons  for the thrid hidden layer of PLNN model  
  
-epochs : set the epochs for training a PLNN model  
  
-result_path : set the ouput file for saving shapelets set founded
