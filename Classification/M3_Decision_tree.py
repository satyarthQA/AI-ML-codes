#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[3]:


data_set=pd.read_csv("/Users/satyarth/Downloads/TCGA-PANCAN-HiSeq-801x20531/data.csv",index_col=[0])
data_set


# In[4]:


data_set.isnull()


# In[5]:


data_set.info()


# In[6]:


data_set.isnull().sum()#here is no missing value


# In[7]:


label_data=pd.read_csv("/Users/satyarth/Downloads/M3_Project_/labels.csv",index_col=[0])
label_data


# In[8]:


frames = [label_data, data_set]
df = pd.concat(frames, axis=1)
df.reset_index(drop=True, inplace=True)

# checking the shape of the combined dataframe
df.shape


# In[9]:


df


# In[10]:


X = df.iloc[:, 1:81]
X


# In[11]:


Y=label_data.values
Y


# In[12]:


from sklearn import preprocessing
from sklearn import utils

#convert y values to categorical values
lab = preprocessing.LabelEncoder()
y_transformed = lab.fit_transform(Y)

#view transformed values
print(y_transformed)



# In[13]:


from sklearn.model_selection import train_test_split
X_train,X_test, Y_train,Y_test= train_test_split(X,y_transformed,test_size=.2,random_state=42)


# In[14]:


X_train


# In[15]:


from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier()
clf.fit(X_train,Y_train)


# In[16]:


Y_pred= clf.predict(X_test)
Y_pred


# In[17]:


from sklearn.metrics import accuracy_score
accuracy=accuracy_score(Y_test,Y_pred)
accuracy


# In[18]:


from sklearn.ensemble import RandomForestClassifier
my_model = RandomForestClassifier(n_estimators = 100, random_state=50)

# Train the model
my_model.fit(X_train, Y_train)


# In[19]:


y_predict_r=my_model.predict(X_test)
y_predict_r


# In[20]:


accuracy=accuracy_score(Y_test,y_predict_r)
accuracy


# In[21]:


x_norm_train = (X_train-np.min(X_train))/(np.max(X_train)-np.min(X_train))
x_norm_train


# In[22]:


x_test_norm=(X_test-np.min(X_test))/(np.max(X_test)-np.min(X_test))
x_test_norm


# In[23]:


x_norm_train_new=x_norm_train.replace(np. nan,0) 


# In[24]:


x_test_norm_new=x_test_norm.replace(np. nan,0) 


# In[25]:


from sklearn.tree import DecisionTreeClassifier
clf_new=DecisionTreeClassifier()
clf_new.fit(x_norm_train_new,Y_train)


# In[26]:


Y_pred_new= clf.predict(x_test_norm_new)
Y_pred_new


# In[27]:


accuracy=accuracy_score(Y_test,Y_pred_new)
accuracy


# In[28]:


from sklearn.ensemble import RandomForestClassifier
my_model_random_scale = RandomForestClassifier(n_estimators = 100, random_state=50)

# Train the model
my_model_random_scale.fit(x_norm_train_new,Y_train)


# In[29]:


y_predict_r_scale=my_model.predict(X_test)
y_predict_r_scale


# In[30]:


accuracy_scale=accuracy_score(Y_test,y_predict_r_scale)
accuracy_scale


# In[44]:


from sklearn.ensemble import BaggingClassifier,RandomForestClassifier
n_feat=X_train.shape[1]


# In[47]:


max_depth=2
n_trees=1


# In[50]:


for i in range(10):
    seed= np.random.randint(low=0,high=100)# we want result reproducible
    df=DecisionTreeClassifier(criterion='gini',splitter='best',max_depth=max_depth,random_state=seed)
    df.fit(X_train, Y_train)
    y_dt_pred=df.predict(X_test)
    accuracy_dt=accuracy_score(Y_test,y_dt_pred)
    print("accuracy of decision tree:",accuracy_dt)
    
    bagging=BaggingClassifier(base_estimator=DecisionTreeClassifier(criterion='gini',splitter='best',max_depth=max_depth,random_state=seed),
    n_estimators=n_trees,max_features=int(n_feat),bootstrap=True,bootstrap_features=False,oob_score=True,random_state=seed)
    bagging.fit(X_train,Y_train)
    Y_pred_bag=bagging.predict(X_test)
    accuracy_bagging=accuracy_score(Y_test,Y_pred_bag)  
    print("accuracy of bagging:",accuracy_bagging)
    
    random_forest=RandomForestClassifier(n_estimators=n_trees,criterion='gini',max_depth=max_depth,max_features=int(n_feat**0.5),oob_score=True,random_state=seed)
    random_forest.fit(X_train,Y_train)
    Y_pred_random=random_forest.predict(X_test)
    accuracy_random=accuracy_score(Y_test,Y_pred_random)  
    print("accuracy of bagging:",accuracy_random)


# In[53]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score


# Initialize lists to store accuracy values
acc_decision_tree = []
acc_random_forest = []
acc_boosting = []

# Vary the number of trees
num_trees_range = range(1, 101)

# Loop over different numbers of trees
for num_trees in num_trees_range:
   # Decision Tree
   dt_model = DecisionTreeClassifier()
   dt_model.fit(X_train, Y_train)
   dt_pred = dt_model.predict(X_test)
   acc_decision_tree.append(accuracy_score(Y_test, dt_pred))
   
   # Random Forest
   rf_model = RandomForestClassifier(n_estimators=num_trees)
   rf_model.fit(X_train, Y_train)
   rf_pred = rf_model.predict(X_test)
   acc_random_forest.append(accuracy_score(Y_test, rf_pred))
   
   # Gradient Boosting
   gb_model = GradientBoostingClassifier(n_estimators=num_trees)
   gb_model.fit(X_train, Y_train)
   gb_pred = gb_model.predict(X_test)
   acc_boosting.append(accuracy_score(Y_test, gb_pred))

# Plot the graph
plt.figure(figsize=(10, 6))
plt.plot(num_trees_range, acc_decision_tree, label='Decision Tree')
plt.plot(num_trees_range, acc_random_forest, label='Random Forest')
plt.plot(num_trees_range, acc_boosting, label='Gradient Boosting')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Number of Trees')
plt.legend()
plt.grid(True)
plt.show()


# In[54]:


# Initialize lists to store accuracy values
acc_decision_tree = []
acc_random_forest = []
acc_boosting = []

# Vary the max_features parameter
max_features_range = range(1, X.shape[1] + 1)

# Loop over different max_features values
for max_features in max_features_range:
    # Decision Tree
    dt_model = DecisionTreeClassifier(max_features=max_features)
    dt_model.fit(X_train, Y_train)
    dt_pred = dt_model.predict(X_test)
    acc_decision_tree.append(accuracy_score(Y_test, dt_pred))
    
    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, max_features=max_features)
    rf_model.fit(X_train, Y_train)
    rf_pred = rf_model.predict(X_test)
    acc_random_forest.append(accuracy_score(Y_test, rf_pred))
    
    # Gradient Boosting
    gb_model = GradientBoostingClassifier(n_estimators=100, max_depth=3, subsample=0.8)
    gb_model.fit(X_train, Y_train)
    gb_pred = gb_model.predict(X_test)
    acc_boosting.append(accuracy_score(Y_test, gb_pred))

# Plot the graph
plt.figure(figsize=(10, 6))
plt.plot(max_features_range, acc_decision_tree, label='Decision Tree')
plt.plot(max_features_range, acc_random_forest, label='Random Forest')
plt.plot(max_features_range, acc_boosting, label='Gradient Boosting')
plt.xlabel('max_features')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. max_features')
plt.legend()
plt.grid(True)
plt.show()


# In[58]:


# Train a Decision Tree classifier
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, Y_train)

# Get feature importances
feature_importances = dt_model.feature_importances_

# Sort features in decreasing order of importance
sorted_indices = np.argsort(feature_importances)[::-1]
sorted_importances = feature_importances[sorted_indices]
#sorted_features = data_set.feature_names[sorted_indices]
sorted_features = X.columns[sorted_indices]
# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.bar(range(X.shape[1]), sorted_importances)
plt.xticks(range(X.shape[1]), sorted_features, rotation=45, ha="right")
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importances from Decision Tree Classifier')
plt.tight_layout()
plt.show()

Dimensionality reduction algorithms like Principal Component Analysis (PCA) can help if 
we are dealing with a high-dimensional dataset. PCA reduces the 
dimensionality of our dataset while retaining as much variance as possible. 
It can potentially help by removing noise and focusing on the most important components.
# In[ ]:




