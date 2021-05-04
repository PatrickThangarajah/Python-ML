#!/usr/bin/env python
# coding: utf-8

# # Mech 305 Tutorial Asssignment 8- Patrick Thangarajah

# ## Step 1: Framing the Problem
#     
#    Using the data from a list of titanic passengers and whether they survived, I want to build a machine learning model to predict if someone survives based on their age, class spouses/siblings aboard, parents/children aboard, fare and gender/sex.

# In[2]:


import numpy as np # for linear algebra
import pandas as pd # for dealing data files
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt # for plotting


# ## Step 2: Getting the Data
# The titanic csv was used to get the data, and the age was converted to a float value. The data was split into training data and testing data.

# In[3]:


data = pd.read_csv(r"C:\Users\ptemm\Downloads\titanic.csv");
m = len(data) # 400
data[['Age','Fare','Pclass']] = data[['Age','Fare','Pclass']].astype(float)


# In[4]:


# scikit learn
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(data, test_size=0.2)
train_data.head()


# ## Step 3: Explore the Training Data
# The training data is visualized and explored below.
# 

# In[5]:


train_data.info()


# In[6]:


train_data.describe()


# The names of the passengers are dropped from the set, as it is not used in predicting the survival of the passengers. The survival of the passengers is also dropped from the set as it is not a feature.

# In[7]:


Y_train = train_data["Survived"].copy()
X_train = train_data.drop(["Survived", "Name"], axis=1) 
m_train = len(X_train)
X_train


# In[8]:


train_data["Sex"].value_counts()


# There are 453 males in this training set and 256 females, as shown above. The passenger class counts are shown below. In this data set there 392 3rd class passengers, 170 1st class pasengers, and 147 2nd class passengers.

# In[9]:


train_data["Pclass"].value_counts()


# In[31]:


def plot_scatter_with_labels(X1, X2, Y, xlabel="Age", ylabel="Gender/Sex"): 
    df = pd.DataFrame(dict(x1=X1, x2=X2, label=Y))
    groups = df.groupby("label")
    for name, group in groups:
        plt.plot(group.x1, group.x2, 
                 marker="o", linestyle="", ms=3,
                 label=name)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

plot_scatter_with_labels(train_data["Age"],
                         train_data["Sex"], 
                         train_data["Survived"])


# The plot above shows the gender/sex of the passengers, their age and whether they survived. 
# From the graph for males it is visible that the proportion of people surviving decreases with increasing age.

# In[11]:


def plot_scatter_with_labels(X1, X2, Y, xlabel="Age", ylabel="Passenger Class"): 
    df = pd.DataFrame(dict(x1=X1, x2=X2, label=Y))
    groups = df.groupby("label")
    for name, group in groups:
        plt.plot(group.x1, group.x2, 
                 marker="o", linestyle="", ms=3,
                 label=name)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

plot_scatter_with_labels(train_data["Age"],
                         train_data["Pclass"], 
                         train_data["Survived"])


# The above graph shows the age of the passengers, their class and whether they surivived. From the graph,
# is noticed that there is a lower proportion of thrid class survivors than 1st and 2nd class.

# ## Step 4: Pre-Processing the Data
# 
# The data with words, such as the gender/sex, is converted to numbers. 
# The data is then standardized and the features are scaled.
# Logistic regression is then used to determine which features are the most useful in predicting survival.

# In[12]:


from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
Gender_encoded = ordinal_encoder.fit_transform(X_train[["Sex"]])
Gender_encoded[:5]

X_train["Sex"] = Gender_encoded
X_train


# In[13]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X_train[["Sex","Age","Siblings/Spouses Aboard","Parents/Children Aboard","Fare","Pclass"]])

import statsmodels.api as sm
import patsy as pa
from patsy import dmatrices

logit = sm.Logit(Y_train,X)
logit.fit().params


# From this result:
# - x1- Sex
# - x2- Age
# - x3- Siblings/Spouses Aboard
# - x4- Parents/Children Aboard
# - x5- Fare
# - x6- Passenger Class
# 
# Age, Sex, and Passenger class have larger coefficients in the logistic regression, so they are useful in predicting a passengers chances of surviving. These features will be used in the Machine Learning Models 

# In[14]:


X_train = scaler.fit_transform(X_train[["Age","Sex","Pclass"]])


# ## Step 5: Applying Machine Learning Models
# 
# Machine Learning Models are used with the age, sex and passenger class as the features. Below each model is the average cross-validation scores. Since there are three features, plots could not be used.

# ### Method 1: Discriminant Analysis Classifier

# In[15]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
da_clf = LinearDiscriminantAnalysis()
da_clf.fit(X_train, Y_train)


# In[16]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(da_clf, X_train, Y_train,
                        scoring="accuracy", cv=10)
scores


# In[17]:


print('Cross-Validation Score:')
np.average(scores) # averaged cv scores of 10 cross validations


# In[18]:


avg_cv_scores = {} # dictionary to store all avg scores
avg_cv_scores["DA"] = np.average(scores)


# In[19]:


axes =[-3,3,-3,3,-3,3]

x0s = np.linspace(axes[0], axes[1], 100)
x1s = np.linspace(axes[2], axes[3], 100)
x2s = np.linspace(axes[4], axes[5], 100)
x0, x1, x2 = np.meshgrid(x0s, x1s, x2s)
X = np.c_[x0.ravel(), x1.ravel(),x2.ravel()]
y_pred = da_clf.predict(X)
y_pred
#y_pred = y_pred == "Survived"
#if inverse:
#        X = scaler.inverse_transform(np.c_[x0s, x1s])
#        x0, x1 = np.meshgrid(X[:,0], X[:,1])


# ### Model 2: Naive Bayes Classifier

# In[20]:


from sklearn.naive_bayes import GaussianNB
nb_clf = GaussianNB()
nb_clf.fit(X_train, Y_train)
scores = cross_val_score(nb_clf, X_train, Y_train,
                        scoring="accuracy", cv=10)
avg_cv_scores["NB"] = np.average(scores)
print('Cross-Validation Score:')
print(avg_cv_scores["NB"])


# ### Model 3: KNN

# In[21]:


from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_neighbors=6) #6 gave the best score 
knn_clf.fit(X_train, Y_train)
scores = cross_val_score(knn_clf, X_train, Y_train,
                        scoring="accuracy", cv=10)
avg_cv_scores["KNN3"] = np.average(scores)
print('Cross-Validation Score:')
print(avg_cv_scores["KNN3"] )


# ### Model 4: SVM-Linear

# In[22]:


from sklearn.svm import SVC
svmln_clf = SVC(kernel="linear", C=1)
svmln_clf.fit(X_train, Y_train)
scores = cross_val_score(svmln_clf, X_train, Y_train,
                        scoring="accuracy", cv=10)
avg_cv_scores["SVM-linear"] = np.average(scores)
print('Cross-Validation Score:')
print(avg_cv_scores["SVM-linear"] )


# ### Model 5:SVM-Non-linear

# In[23]:


svmpoly_clf = SVC(kernel="poly", gamma='auto') 
svmpoly_clf.fit(X_train, Y_train)
scores = cross_val_score(svmpoly_clf, X_train, Y_train,
                        scoring="accuracy", cv=10)
avg_cv_scores["SVM-poly"] = np.average(scores)
print('Cross-Validation Score:')
print(avg_cv_scores["SVM-poly"] )


# In[24]:


#Gaussian Radial Basis Function (RBF) Kernel
svmgauss_clf = SVC(kernel="rbf", gamma=1, C=1) 
svmgauss_clf.fit(X_train, Y_train)
scores = cross_val_score(svmgauss_clf, X_train, Y_train,
                        scoring="accuracy", cv=10)
avg_cv_scores["SVM-gauss"] = np.average(scores)
print('Cross-Validation Score:')
print(avg_cv_scores["SVM-gauss"] )


# ## Step 6:Fine-Tuning the System
# 
# The cross validation scores for each model are shown below.

# In[25]:


avg_cv_scores


# From the scores, the  KNN3 Model had the best score, so it will be used.
# The perfomance on the test set is checked and shown below.

# In[26]:


from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
Gender_encoded = ordinal_encoder.fit_transform(test_data[["Sex"]])
Gender_encoded[:5]

test_data["Sex"] = Gender_encoded


# In[27]:


# Note you need to use "transform" not "fit_transform". 
# This will standardize the test_data using the mean & std of the train_data.
#test_data["Sex"] = Gender_encoded

X_test = scaler.transform(test_data[["Age", "Sex","Pclass"]])
Y_test_pred = knn_clf.predict(X_test)
accuracy = np.sum(Y_test_pred==test_data["Survived"])/len(Y_test_pred)
print("Accuracy on test set: ",accuracy)


# ## Step 7: Solution  and Launch of Final Model

# The classifier can predict 84% of whether an individual surivives the sinking of the titanic based on their age, gender/sex, and class. 
# The gender/sex and age of an individual plays a significant role of whether someone would survive the titanic, as the lifeboat boarding process prioritized women and children first. Additonally, class also had a role in determining survival, as individuals from a high class(1st class) were more likely to reach a life boat.

# In[28]:


from sklearn.pipeline import Pipeline

full_pipeline = Pipeline([
    ('std_scaler', StandardScaler()),
    ('knn_clf', SVC(kernel="rbf", gamma=1, C=1)),
])
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
Gender_encoded = ordinal_encoder.fit_transform(data[["Sex"]])
Gender_encoded[:5]
data["Sex"]=Gender_encoded

full_pipeline.fit(data[["Age", "Sex","Pclass"]], data["Survived"])


# Using the best classifiers, a pipeline and a final machine learning model is created.
# This KNN model uses the sex, age, and passenger class data to predict whether an individual would survive the sinking of the titanic.
# 
# 
# The model is then used to predict whether the following individuals survive
# - a 5 year old male 3rd class passenger
# - a 10 year old male 2nd class passenger
# - a 7 year old male 1st class passenger
# - a 21 year old male 1st class passenger
# - a 22 year old female 3rd class passenger
# - 48 year old female 1st class passenger
# 
# The 0 under Sex means the individual is a male, and the 1 means that the individual is a female.
# 
# 

# In[29]:


X_passengers = pd.DataFrame({'Age':  [5.,10.,7.,21.,22., 48.], 'Sex': [1.,1.,1.,1.,0.,0.],'Pclass':[3.,2.,1.,1.,3.,1.]})
X_passengers


# In[30]:


full_pipeline.predict(X_passengers)


# The above array shows the survival of each individal, and corresponds to each row on the table above.
# In the array 0 means they did not survive,and 1 means they did survive.
# Based on the model, the following individuals survived:
# 
# - 10 year old 1st class male
# - 48 year old 1st class female 
# - 22 year old 3rd class female 
# - 7 year old  1st class male
# 
# This prediction makes sense, as the 1st class individuals generally had a higher chance of surivival 
# than the lower class individuals due to their location on the boat and boarding procedures. The passenger 
# class also has a higher impact on predictions than age, as the 7 year old 1st class male and 10 year old 2nd class male survived, but the 5 year old 3rd class male didnt survive. Additionally, females were prioritized in filling the lifeboats, which is why the two females survived regardless of class.
