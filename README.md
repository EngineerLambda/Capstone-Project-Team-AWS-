# Capstone-Project-Team-AWS-

# ** Fake BankNote Detection. **

The society is dealing with a huge number of counterfeit money as a result of the significant technological advancement, which ranges from bills with poor printing quality to those with excellent quality. Despite the extensive use of electronic and online payments, most people still find it necessary to use cash. Also, the idea of electronic payments plays a significant part in the fight against money forgery because these transactions are significantly safer because governments can monitor the source of the money. 

Given this problem, it is essential to accurately and effectively identify fake currency and tell them apart from real currency. As is well known, manual detection requires more time, is slower, and is less effective than automatic. 

This work employed various machine learning in order to get the most accurate prediction for our data. Here we will investigate 7 distinct models for the Fake Bank Notes Detection project, with Gradient Boosting & XG Boosting making up two of them. In response to Kearns and Valiant's inquiry, some of the questions we hope to answer are, “Can a set of weak learners create a single strong learner?," With the recent development of XG Boosting, we employed this ML tool which works by running several weak classifiers sequentially to provide the boosting effect. Following the computation of an output by one of the classifiers, the model will calculate the error before moving on to the next classifier. Unless the model parameters tell it to stop, it will repeat this cycle. This should enable it to produce outputs that are highly accurate and polished. This contrasts with bagging, which runs several distinct models concurrently and asks the participants to vote on which result is the correct one. Boosting is quite accurate and doesn't require a lot of hypertuning. However, it has the drawback of being computationally expensive and consuming a lot of power. Due to this, the model has the propensity to take a while. 

Random Forest was also employed. The program creates numerous decision trees and outputs the mean/mode of each tree's forecast. One advantage of Random Forest is that you can use it to classify objects quite accurately. Moreover, it can handle big datasets and guards against overfitting. Random Forests, on the other hand, are challenging to interpret.

In order to take in data and utilize that data to forecast the results of future data, the logistic regression method was devised. To forecast how many bills would be fake and how many would be legitimate, we employed logistic regression in our model.

We chose to use Random Forest since it outperformed Logistic Regression, Support Vector Machine (SVM/SVC), K-Nearest Neighbors (KNN), Decision Tree, Boosting, and Support Vector Machine in terms of accuracy for the following reasons: 

- Compared to other models like Support Vector Machines or Neural Networks, Random Forests are easier to understand. In order to comprehend the problem domain and effectively communicate the findings to stakeholders, it is helpful for the Random Forest algorithm to be able to reveal which features are significant in producing the predictions. 
- Random Forest is typically quicker when dealing with huge datasets than certain alternative models, such as K-Nearest Neighbors or Boosting. 
- The Random Forest technique is capable of handling noisy input and missing values. Moreover, the technique is not overly sensitive to how the features are scaled, making it simpler to apply to datasets with varied feature sizes.
- Lastly, it performs well with high-dimensional data and can handle a large number of input features. Because of this, data scientists that work with complicated datasets should consider using it.
