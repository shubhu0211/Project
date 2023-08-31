#!/usr/bin/env python
# coding: utf-8

# # Objective for classification project on Diabetes dataset: 
# 
# - Develop a machine learning classification model to predict the presence or absence of diabetes in patients based on a set of input features. The goal of this project is to create an accurate and reliable model that can assist healthcare professionals in early diagnosis and proactive management of diabetes. 
# 
# 
# - The dataset consists of various demographic, clinical, and lifestyle factors for a diverse group of patients, along with a binary label indicating the presence (1) or absence (0) of diabetes. Your task is to explore the data, preprocess it if necessary, and employ various classification algorithms to train and evaluate the model's performance. 
# 
# 
# - In addition to achieving a high accuracy rate, you should pay attention to other important evaluation metrics such as precision, recall, and F1-score to ensure the model's ability to correctly identify diabetic patients while minimizing false positives and false negatives. 
# 
# 
# 
# - To accomplish the objective, consider the following steps:
# 1. Data Preprocessing: Handle missing values, normalize/standardize features, and perform any necessary data transformations.
# 2. Feature Selection: Identify relevant features that contribute most to the model's predictive power and remove any irrelevant or redundant ones.
# 3. Model Selection: Compare and contrast different classification algorithms (e.g., logistic regression, decision trees, random forests, support vector machines, etc.) to find the most suitable one for this particular task.
# 4. Model Training and Evaluation: Split the data into training and testing sets, train the selected model(s), and evaluate their performance using appropriate metrics.
# 5. Hyperparameter Tuning: Optimize the chosen model's hyperparameters to further improve its performance.

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[53]:


df=pd.read_csv('pima_diabities.csv')


# In[3]:


df.head()


# # 

# # EDA process
# - The Exploratory Data Analysis (EDA) process is a crucial initial step in machine learning projects that involves examining and understanding the data before building predictive models. EDA serves several important purposes and offers various benefits in the machine learning workflow:
# 
# 1. Data Understanding: EDA helps data scientists and machine learning practitioners gain insights into the dataset's structure, size, and content. It allows them to familiarize themselves with the variables, their types, and potential relationships between them.
# 
# 2. Data Cleaning: During the EDA process, data inconsistencies, missing values, outliers, and noise can be identified and addressed. Data cleaning is essential for building reliable and accurate machine learning models.
# 
# 3. Data Visualization: EDA often involves the use of various visualizations, such as histograms, scatter plots, box plots, and correlation matrices. Data visualization makes patterns and trends more accessible, making it easier to detect relationships between variables and spot potential problems.
# 
# 4. Data Preparation: EDA provides insights into the distribution of data, which can influence how data is preprocessed and normalized or standardized for model training. Understanding the data distribution helps select appropriate data transformation techniques.
# 
# 5. Outlier Detection: Outliers, which are extreme values that deviate significantly from the rest of the data, can impact model performance. EDA helps in identifying outliers, which can be handled appropriately, either by removing, transforming, or handling them during preprocessing.
# 
# 
# ### 1) checking datatypes of the columns in the data

# In[4]:


df.dtypes


# ### 2) Changing object datatype into integer
# - changing data types is an essential step for several reasons. Data type conversion and manipulation help prepare the dataset for better analysis and model building.

# In[5]:


df['mass']=df['mass'].astype('int64')
df['pedi']=df['pedi'].astype('int64')


# In[6]:


df.dtypes


# ### 3) Describe use to show 5 point summery of the data

# In[7]:


df.describe().T  #use to show 5 point summery


# ### 4) checking null values in the data and this data dont have any null values

# In[8]:


df.isnull().sum()


# ### 5) here in below step, showing the skewness of the data

# In[9]:


df.skew()


# ### 6) plotting pie chart for having diabetes and not having diabetes 

# In[75]:


plt.pie(df['class'].value_counts(normalize=True)*100,labels=['not having dia','having dia'],autopct='%.1f%%')
plt.show()


# ### 7) In below step, plot the boxplot for detecting outliers 

# In[10]:


plt.figure(figsize=(12,10))
df.boxplot()
plt.show()


# ### 8) In below step, 1st. showing the correlation of tha data and then plotting heatmap to show correleation in visualization method

# In[11]:


df.corr()


# In[12]:


plt.figure(figsize=(12,10))
sns.heatmap(df.corr(),annot=True)
plt.show()


# # 

# # Model building for classification

# In[13]:


x=df.drop(['class'],axis=1)
y=df['class']


# In[14]:


from sklearn.model_selection import train_test_split


# In[15]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=41)


# In[16]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# # 

# ## 1) Logistic regression model and applying classification evaluation metrics
# - Logistic Regression is a statistical technique used for binary classification tasks, where the target variable can take only two discrete values (usually represented as 0 and 1). In its name, logistic regression is a classification algorithm, not a regression algorithm.

# In[17]:


from sklearn.linear_model import LogisticRegression


# In[18]:


lr=LogisticRegression()
lr.fit(x_train,y_train)
y_true,y_pred=y_test,lr.predict(x_test)


# #### i) Importing the evaluation matrix of classification
# - evaluation metrics are used to assess the performance of a model by comparing its predictions to the actual class labels in the dataset.

# In[19]:


from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve,precision_score,recall_score,f1_score


# #### Training accuracy of data
# - Training accuracy and testing accuracy are both metrics used to evaluate the performance of a machine learning model, particularly in a classification task. These metrics provide insights into how well the model is performing during different stages of the learning process.

# In[20]:


lr.score(x_train,y_train)*100  #training accuracy


# #### Testing accuracy of data

# In[21]:


lr.score(x_test,y_test)*100  #testing accuracy


# #### a) Total accuracy of the data by using accuracy matrix
# - Accuracy measures the proportion of correctly classified instances (both true positives and true negatives) out of the total number of instances. It is the most straightforward metric but can be misleading if the classes are imbalanced.

# In[22]:


accuracy_score(y_true,y_pred)*100  #total accuracy of your data


# #### b) To check performance of test data by using Confusion matrix
# -  A confusion matrix is a tabular representation of the model's predictions versus the actual class labels. It shows the number of true positives, true negatives, false positives, and false negatives. It is a useful tool for understanding the model's performance on each class.

# In[23]:


sns.heatmap(confusion_matrix(y_true,y_pred),annot=True)
plt.title('Confusion Matrix')
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.show()


# #### c) The ROC AUC curve:
# - The ROC curve plots the true positive rate (recall) against the false positive rate at various classification thresholds. AUC-ROC measures the overall performance of the model, and an AUC value closer to 1 suggests a better-performing model.

# In[24]:


tnr,tpr,_=roc_curve(y_true,y_pred)
plt.plot(tnr,tpr)
plt.title('ROC - AUC curve')
plt.xlabel('Specificity')
plt.ylabel('Sensitivity')
plt.show()


# #### d) Below step showing Precision, Recall and F1.score of the classification data
# - Precision measures the proportion of true positive predictions (correctly predicted diabetics) out of all instances predicted as positive (both true positives and false positives). High precision indicates that when the model predicts a patient as diabetic, it is likely to be correct.
# 
# 
# - Recall calculates the proportion of true positive predictions (correctly predicted diabetics) out of all actual positive instances (true positives and false negatives). High recall indicates that the model is capable of identifying most of the actual diabetic patients.
# 
# 
# - The F1-score is the harmonic mean of precision and recall, providing a balance between the two metrics. It is useful when precision and recall have different priorities, as it considers both false positives and false negatives. F1-score is a good metric to assess the overall performance of the model, especially when dealing with imbalanced datasets.

# In[25]:


precision_score(y_true,y_pred)*100


# In[26]:


recall_score(y_true,y_pred)*100


# In[27]:


f1_score(y_true,y_pred)*100


# # 

# ## 2) KNN classification
# - K-Nearest Neighbors (KNN) is a non-parametric classification and regression algorithm used for both supervised learning tasks. KNN is a simple algorithm. in KNN, "k" is a user-defined parameter.
# 
# 
# - A higher accuracy score indicates that the KNN algorithm is more in correctly predicting the class labels of the test instances.

# In[28]:


from sklearn.neighbors import KNeighborsClassifier


# In[29]:


knn=KNeighborsClassifier(n_neighbors=5,weights='distance')
knn.fit(x_train,y_train)
y_true,y_pred=y_test,knn.predict(x_test)


# In[30]:


knn.score(x_test,y_test)*100


# # 

# ## 3) Support Vector Machine
# - The primary goal of SVM is to find the optimal hyperplane that best separates data points belonging to different classes in a high-dimensional feature space.
# 
# 
# - In a binary classification setting, SVM aims to find the hyperplane that maximizes the margin between the two classes. 
# 

# In[31]:


from sklearn.svm import SVC


# In[33]:


svc=SVC(C=0.1,kernel='linear')
svc.fit(x_train,y_train)
y_true,y_pred=y_test,svc.predict(x_test)


# # 

# ## 4) Decision Tree
# - It is a hierarchical tree-like structure from the training data, where each internal node represents a decision based on a feature, each branch represents an outcome of the decision, and each leaf node represents a class label.

# In[54]:


from sklearn.tree import DecisionTreeClassifier


# In[55]:


dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
y_true,y_pred=y_test,dt.predict(x_test)


# In[56]:


dt.score(x_train,y_train)*100


# In[57]:


dt.score(x_test,y_test)*100


# # 

# ### a) In this scenario, the high training score (100) suggests that the model is performing exceptionally well on the test data, achieving a perfect classification accuracy. However, the low training score (63) indicates that the model is not performing as well on the training data itself. This is a sign of overfitting.
# 
# ### b) Pruning: Pruning is a technique that removing branches from the tree that do not significantly contribute to improving model accuracy. This helps simplify the Decision Tree and reduce its complexity, thus preventing it from overfitting.
# 
# 
# - Limiting Tree Depth: Setting a maximum depth for the Decision Tree pruning the number of levels in the tree

# # 

# In[40]:


from sklearn.tree import plot_tree


# #### i) Decision trees use feature names, for example, "age," "income," "gender," etc., to represent the conditions used for splitting the data during the tree-building process.

# In[41]:


fn=list(x_train) 


# #### ii) The class_names parameter is used to assign meaningful labels to the class labels (output categories) in the decision tree. For example, "positive," "negative," "spam," "not spam," etc.

# In[ ]:


cn=['not having diabetes','having diabetes']


# ### c) in below, using limiting the depth for pruning branches of desicion tree
# - the criterion parameter is used to specify the quality measure used for making splits during the tree construction process. The Gini impurity as the criterion to evaluate the quality of potential splits.

# In[68]:


dt1=DecisionTreeClassifier(criterion='gini',max_depth=2)
dt1.fit(x_train,y_train)
y_true,y_pred=y_test,dt1.predict(x_test)


# In[69]:


dt1.score(x_train,y_train)*100


# In[70]:


dt1.score(x_test,y_test)*100


# In[74]:


fig,axes=plt.subplots(figsize=(6,6),dpi=300) #dpi for dot per inch. for measurement of resolution to show decision tree
plot_tree(dt1,feature_names=fn,class_names=cn,filled=True) #fillled use for colored nodes of tree
plt.show()


# In[ ]:




