#!/usr/bin/env python
# coding: utf-8

# Forecasting Readmissions to Hospitals

# Predicting hospital readmissions in a 30-day window is the problem.
# Creating a predictive model specifically for patients at high risk.
# 
# **Assignments:**
# * Preparing Data
# * Feature Engineering,
# * Model Construction, and 
# * Model Assessment

# In[9]:


import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

import matplotlib.pyplot as plt
import seaborn as sns


# In[11]:


import warnings
warnings.filterwarnings("ignore")


# In[13]:


raw_df = pd.read_csv("hospital_readmissions.csv")


# In[15]:


raw_df.head()


# In[ ]:


# Shape


# In[17]:


raw_df.shape


# In[ ]:


#info


# In[19]:


raw_df.info()


# In[ ]:


#null value


# In[21]:


raw_df.isnull().sum()


# In[ ]:


#unique


# In[23]:


raw_df.nunique()


# In[25]:


new = raw_df.drop(columns="Patient_ID", axis=1)
for column in new.columns:
    unique_values = raw_df[column].unique()
    print(f"'{column}':\n {unique_values}\n")


# **Managing Missing Values**

# In[27]:


#Replace null values with na values
raw_df['A1C_Result'].fillna('Unknown', inplace=True)


# In[29]:


raw_df.isnull().sum()


# **Managing Categorical Features**

# In[31]:


raw_df["Gender"]= LabelEncoder().fit_transform(raw_df["Gender"])

raw_df["Gender"].unique()


# In[33]:


raw_df["Admission_Type"]= LabelEncoder().fit_transform(raw_df["Admission_Type"])

raw_df["Admission_Type"].unique()


# In[35]:


raw_df["Diagnosis"]= LabelEncoder().fit_transform(raw_df["Diagnosis"])

raw_df["Diagnosis"].unique()


# In[37]:


raw_df["A1C_Result"]= LabelEncoder().fit_transform(raw_df["A1C_Result"])

raw_df["A1C_Result"].unique()


# In[39]:


raw_df["Readmitted"]= LabelEncoder().fit_transform(raw_df["Readmitted"])

raw_df["Readmitted"].unique()


# In[41]:


raw_df.head()


# **Managing Different Data Types**

# In[43]:


raw_df.info()


# **Managing Redundant Values**

# In[45]:


raw_df.duplicated().sum()


# **Dataframe saved**

# In[47]:


# Saving the Dataframe
raw_df.to_csv("hospital_readmissions_only_int.csv", index= False)


# **Reading the Dataframe**

# In[49]:


df_1 = pd.read_csv("hospital_readmissions_only_int.csv")


# In[51]:


df_1.head()


# **Determining the Unknown(2) values in A1C_Result**

# In[53]:


# Remove rows if the value of 'A1C_Result' is 'Unknown(2)'
A1C_Not_Null = df_1[df_1['A1C_Result'] != 2]


# In[55]:


A1C_Not_Null.head()


# In[57]:


A1C_Not_Null['A1C_Result'].unique()


# In[59]:


A1C_Not_Null.shape


# **Managing Outliers**

# In[ ]:


#Outlier detection with Boxplot


# In[61]:


# Use of function for the box plot    
def plot_box_plots(df, cols):

    plt.figure(figsize=(10, 12))
    
    for i, col in enumerate(cols):
        plt.subplot(4, 4, i + 1)
        sns.boxplot(y=df[col])
        plt.title(col)
    plt.tight_layout()
    plt.show()


# In[63]:


columns = A1C_Not_Null.columns
plot_box_plots(A1C_Not_Null, columns)


# **Use IQR to find the outlier**

# In[65]:


# Compute the IQR and quartiles.
Q1 = A1C_Not_Null.quantile(0.25)
Q3 = A1C_Not_Null.quantile(0.75)
IQR = Q3 - Q1

# Determine the top and lower boundaries for outliers.
upper_bound = Q3 + 1.5 * IQR
lower_bound = Q1 - 1.5 * IQR

# Determining outliers
outliers = A1C_Not_Null[(A1C_Not_Null < lower_bound) | (A1C_Not_Null > upper_bound)]

# Determining outliers
num_outliers = outliers.count()

print("Number of outliers:")
print(num_outliers)


# **Z-score to identify an outlier**

# In[67]:


df_age = A1C_Not_Null["Age"]


# In[69]:


import numpy as np
outliers = []
def detect_outliers_zscore(data):
    thres = 3
    mean = np.mean(data)
    std = np.std(data)
   
    for i in data:
        z_score = (i-mean)/std
        if (np.abs(z_score) > thres):
            outliers.append(i)
    return outliers# driving code

sample_outliers = detect_outliers_zscore(df_age)
print("Outliers from Z-scores method: ", sample_outliers)


# In[71]:


#  changeing every value above the upper threshold to the value of the upper threshold.
#changeing every value below the lower threshold to the value of the lower threshold.
def outlier(df,col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3-Q1

    upper_threshold = Q3 + (1.5*IQR)
    lower_threshold = Q1 - (1.5*IQR)

    df["Age_New"] = df[col].clip(lower_threshold, upper_threshold)


# In[73]:


outlier(A1C_Not_Null,"Age")


# In[ ]:


#box plot following outlier handling


# In[75]:


plt.figure(figsize=(3, 5))
sns.boxplot(data=A1C_Not_Null, y=A1C_Not_Null["Age_New"])


# In[77]:


A1C_Not_Null.describe().T


# In[162]:


#the 'age' column is dropped


# In[79]:


A1C_Not_Null_1 = A1C_Not_Null.drop(columns=["Age"], axis=1)


# **Managing Skewness**

# In[81]:


# Histogram's function 
def plot_histograms(df, cols):

    plt.figure(figsize=(8, 15))

    for i, col in enumerate(cols):
        plt.subplot(7,2, i+1)
        sns.histplot(df[col],kde= True, bins=30, color="salmon") 
        plt.title(col)
    plt.tight_layout()
    plt.show()


# In[83]:


columns = A1C_Not_Null_1.columns
plot_histograms(A1C_Not_Null_1, columns)


# In[85]:


A1C_Not_Null_1.skew()


# Skewness is a metric for asymmetry.
# The range of skewness values is -1 to 1.
# * In case the skewness falls within the range of -0.5 and 0.5, then the distribution is almost symmetrical.
# * In the event where the skewness is less than -0.5, the distribution is left-skewed, or negatively skewed.
# * Positive skewness, or right-skewed distribution, occurs when the skewness value exceeds 0.5.

# **Selecting Feature**

# In[87]:


# heatmap
plt.figure(figsize=(12,6))
sns.heatmap(A1C_Not_Null_1.corr(), annot=True, cmap="Reds")
plt.show()


# In[89]:


# importing VIF library
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calc_vif(X):

    # Finding the VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)


# In[91]:


calc_vif(A1C_Not_Null_1)


# VIF(Variance Inflation factor):
# * When the VIF score is less than 5, it suggests that multicollinearity is not a serious issue and that there is probably not much correlation between the predictor variables.
# * The range of 5 to 10 in the VIF indicates medium multicollinearity.
# * Values of VIF greater than 10 suggest the possibility of significant multicollinearity.

# In[ ]:


#Removing the unnecessary and multicollinear columns  


# In[93]:


A1C_Not_Null_2 = A1C_Not_Null_1.drop(columns=["Age_New","Patient_ID"], axis=1)


# In[95]:


calc_vif(A1C_Not_Null_2)


# In[97]:


A1C_Not_Null_2.isnull().sum()


# In[99]:


A1C_Not_Null_2.head()


# In[101]:


A1C_Not_Null_2.shape


# In[103]:


# saving with real values of  A1C 

A1C_Not_Null_2.to_csv("hospital_with_actual_A1C.csv", index= False)


# **A1C Value Prediction Model**

# In[105]:


# importing libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, auc, roc_curve, confusion_matrix, classification_report

from imblearn.combine import SMOTETomek

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

import pickle


# In[107]:


A1C_Not_Null_2.columns


# In[109]:


A1C_Not_Null_2["A1C_Result"].value_counts()


# In[111]:


#  Splitting of data

x = A1C_Not_Null_2.drop(columns=["A1C_Result"],axis=1) # the independent variables.
y = A1C_Not_Null_2["A1C_Result"] #the dependent variables


# **Handling Unbalanced Feature -> "SMOTE-Tomek"**
# 
# This technique includes:
# * SMOTE's capacity to produce artificial data for minority classes
# * Tomek can eliminate data from the majority class that are recognized as Tomek links.
# * Tomek linkages are very close pairs of instances from various classes that belong to different classes.

# In[113]:


#smotetomek to balance
x_new, y_new = SMOTETomek().fit_resample(x,y)


# In[115]:


print(len(x_new))
print(len(y_new))


# In[117]:


# Logistic Regression

# splitting data into train & test sets
x_train, x_test, y_train, y_test = train_test_split(x_new, y_new, test_size= 0.2, random_state=40)

model = LogisticRegression(solver='liblinear').fit(x_train, y_train)

y_pred_train = model.predict(x_train)
y_pred_test = model.predict(x_test)

#checking the accuracy_score
accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)

metrics ={"Algorithm": "Logistic Regression",
           "Accuracy_Train": accuracy_train,
           "Accuracy_Test": accuracy_test}
print(metrics)


# In[119]:


# The SVM Classification

# splitting dataset into train & test 
x_train, x_test, y_train, y_test = train_test_split(x_new, y_new, test_size= 0.2, random_state=40)

svm = SVC(kernel="rbf", gamma=0.5, C=1.0)
model = svm.fit(x_train, y_train)

y_pred_train = model.predict(x_train)
y_pred_test = model.predict(x_test)

#checking the accuracy_score
accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)

metrics ={"Algorithm": "SVM",
           "Accuracy_Train": accuracy_train,
           "Accuracy_Test": accuracy_test}
print(metrics)


# In[121]:


# Alternative algorithms for classification

def accuracy_checking(x_data, y_data, algorithm):
    
    # splitting dataset into train & test set
    x_train, x_test, y_train, y_test= train_test_split(x_data, y_data, test_size= 0.2, random_state=50)

    model = algorithm().fit(x_train, y_train)

    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)

    #validating the accuracy_score
    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)

    metrics = {"Algorithm": algorithm.__name__,
               "Accuracy_Train": accuracy_train,
               "Accuracy_Test": accuracy_test}
    return metrics


# In[123]:


print(accuracy_checking(x_new,y_new,DecisionTreeClassifier))
print(accuracy_checking(x_new,y_new,RandomForestClassifier))
print(accuracy_checking(x_new,y_new,ExtraTreesClassifier))
print(accuracy_checking(x_new,y_new,AdaBoostClassifier))
print(accuracy_checking(x_new,y_new,GradientBoostingClassifier))
print(accuracy_checking(x_new,y_new,XGBClassifier))


# **Cross Validation**

# In[125]:


# using KFold Cross Validation
from sklearn.model_selection import cross_val_score, StratifiedKFold

# launching the classification model
A1C_Model = GradientBoostingClassifier()

# Launching K-Fold cross-validator
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Performing K-Fold Cross-Validation and calculating each fold accuracy 
accuracy_scores = cross_val_score(model, x_new, y_new, scoring='accuracy', cv=skf)
mean_accuracy = np.mean(accuracy_scores)

# Printing
print("Accuracy scores for each fold:", accuracy_scores)
print("Mean Accuracy:", mean_accuracy)


# In[127]:


#  Model Selection
x_train, x_test, y_train, y_test= train_test_split(x_new, y_new, test_size= 0.2, random_state= 42)

A1C_Model = GradientBoostingClassifier().fit(x_train, y_train)
 
y_pred_train = model.predict(x_train)
y_pred_test = model.predict(x_test)


# **Performance Measures**

# In[129]:


# train and test  accuracy_score

accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)

print("Accuracy score for Train and Test")
print("----------------------------------")
print("Accuracy_Train: ",accuracy_train)
print("Accuracy_Test: ",accuracy_test)


# In[131]:


# confution metrics 

print("Confution_matrix for Test")
print("--------------------------")
print(confusion_matrix(y_true = y_test, y_pred = y_pred_test))


# In[133]:


# Typically, a classification report contains metrics like support, F1-score, precision, and recall.

print("Classification_report for Test")
print("-------------------------------")
print(classification_report(y_true= y_test, y_pred= y_pred_test))


# In[135]:


# (ROC) Receiver Operating Characteristic Curve

FP, TP, Threshold = roc_curve(y_true=y_test, y_score=y_pred_test)

print(FP)
print(TP)
print(Threshold)


# In[137]:


# Area Under the (AUC) Curve 

auc_curve = auc(x=FP, y=TP)
print("auc_curve: ", auc_curve)


# In[139]:


# Plotting ROC-AUC curve

roc_point= {"ROC Curve (area)":round(auc_curve, 2)}
plt.plot(FP,TP,label= roc_point)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.1])
plt.xlabel("False Positive")
plt.ylabel("True Positive")
plt.plot([0,1],[0,1],"k--")
plt.legend(loc= "lower right")
plt.show()


# In[141]:


# unsing pickle for Saving the Model 
with open("A1C_Model.pkl","wb") as m:
    pickle.dump(A1C_Model, m)


# In[ ]:




