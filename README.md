# Credit_Risk_Analysis
## Overview
This project uses machine learning statistical alghorithms to analyze loan data to evaluate and predict credit risk. The project focuses on supervised learning and we use different machine learning techniques to train and evalute the data with unbalanced classes. The dataser from the LendingClub has an unbalanced classification problem due to the number of good loans outweighing the amount of risky loans, so we need to employ various alghorithms to resample the data to increase meaningful predictions and improve the accuracy scores. These techniques include RandomOverSampler, SMOTE, SMOTEEN, ClusterCentroids, BalancedRandomForestClassifier, and EasyEnsembleClassifier. 
## Results
In this project we need to resample and evalute the dataset using Python libraries scikit-learn and imbalanced-learn. The original dataset contained 115.675 loan applications for Quarter 1 of 2019. We used loan status to determine whether the application was considered low or high risk, whereas applications that had current as the loan status were classified as low risk and the remaining as high risk. This reduced the dataset to 68,817 applications with 99% classified as low risk.
![This is an image](https://github.com/weise142/Credit_Risk_Analysis/blob/main/images/loan%20count%20resampled.png)
We then used the 75%-25% method to split the data for training vs testing creating a dataset with 51,366 low risk and 246 high risk applications for the training set.
![This is an image](https://github.com/weise142/Credit_Risk_Analysis/blob/main/images/train%20test%20data.png)
## Oversampling
First we used the RandomOverSampler model to randomly select from the minority class and add it to the training set until both classifications are equal. The balanced accuracy score was 64%, the high risk precsion rate was 1% with the recall at 66% giving the model a F1 score of 2%, while the low risk had a precision rate of 100% and recall at 62%.
![This is an image](https://github.com/weise142/Credit_Risk_Analysis/blob/main/images/oversample.png)
![This is an image](https://github.com/weise142/Credit_Risk_Analysis/blob/main/images/oversampleclasses.png)
## SMOTE
Next we used the SMOTE technique, which like RandomOverSampler increases the size of the minority class by creating new values based on the value of the closest neighbors to the minority class instead of random selection. The balanced accuracy score improved a bit to 65.1%. The high risk precision rate was 1% with the recall at 61% giving this model a F1 score of 2%, while low risk had a precision rate of 100% and a recall at 69%.
![This is an image](https://github.com/weise142/Credit_Risk_Analysis/blob/main/images/smote.png)
![This is an image](https://github.com/weise142/Credit_Risk_Analysis/blob/main/images/smote3.png)
## Undersampling
Undersampling uses the ClusterCentroids Model to identify clusters of the majority class to generate synthetic data points that are representative of the clusters. This model classified 246 records for each high and low risk. The balanced acuracy score was 54.5%. The high risk precision rate was 1% with the recall at 69% giving the model a F1 score of 1%, while the low risk had a precision rate of 100% and a recall at 40%. 
![This is an image](https://github.com/weise142/Credit_Risk_Analysis/blob/main/images/undersample%20accuracy.png)
![This is an image](https://github.com/weise142/Credit_Risk_Analysis/blob/main/images/underclass.png)
## SMOTEENN
We then used SMOTEENN which combines aspects of both oversampling and undersampling. This model classified 68,460 high risk records and 62,011 low risk records. The balanced accuracy score was 64.5%. The high risk precision rate was 1%, with the recall at 72% giving this model a F1 score of 2%, while the low risk precision rate was 100% and the recall was 57%.
![This is an image](https://github.com/weise142/Credit_Risk_Analysis/blob/main/images/smoteenaccuracy.png)
![This is an image](https://github.com/weise142/Credit_Risk_Analysis/blob/main/images/smoteenclass.png)
## BalancedRandomForestClassifier
Next we look at the Balanced Random Forest Classifier which randomly under-samples each bootstrap sample to balance it. For this model the balanced accuracy score was 78.9%. The high risk precidion rate was 3% with the recall at 70% giving the model a F1 score of 6%, while the low risk had a precision rate of 100% with the recall at 87%. 
![This is an image](https://github.com/weise142/Credit_Risk_Analysis/blob/main/images/Balancedaccuracy.png)
![This is an image](https://github.com/weise142/Credit_Risk_Analysis/blob/main/images/balancedclass.png)
![This is an image](https://github.com/weise142/Credit_Risk_Analysis/blob/main/images/BalancedFeatures.png)
## EasyEnsembleClassifier
Lastly, we use the Easy Ensemble Classifier model which creates an ensemble set by iteratively applying random undersampling. The method iteratively selects a random subset and makes an ensemble of the different set. For this model, the balanced accuracy score was 93.2% . The high risk precision rate was 9% with the recall at 92% giving the model a F1 score of 16%, while the low risk had a precision rate of 100% with the recall at 94%. 
![This is an image](https://github.com/weise142/Credit_Risk_Analysis/blob/main/images/Easyaccuracy.png)
![This is an image](https://github.com/weise142/Credit_Risk_Analysis/blob/main/images/Easyclass.png)
# Summary
After reviewing all of the models, we can clearly see that the Easy Ensemble Classifier model yielded the best results when comparing both the low risk and high risk prediction results. For low risk prediction rates the accuracy was 93.2% and the precision rate was 9%, while the results for the low risk portion of the model yielded a sensitivty rate of 94% and a F1 score of 97%. All of these statistics were the highest amongst all models. I would suggest using this model when attempting to predict credit risk as the largest number of loans are considered low risk so having a high precision rate on low risk loans is most important for these models.
