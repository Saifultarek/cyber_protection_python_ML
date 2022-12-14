#!/usr/bin/env python
# coding: utf-8

# ## **Data set description**
# 01. gender ( Feamle -> 0, Male -> 1 )
# 02. age
# 
# #      Age Group      Value
#    *   13 - 23 -> 0
#    *   24 - 33 -> 1
#    *   34 - 43 -> 2
#    *   44 - 53 -> 3
#    *   54 - 63 -> 4
# 
# 03. occupation
# 
# #     Group  Value
#   *   Student -> 3
#   *   Housewife -> 1
#   *   Doctor -> 0
#   *   Teacher -> 4
#   *   Other -> 5
# 
# 04. idea_about_cb  -> Do you have any idea about Cyberbullying? ( Yes - 1, No - 0)
# 05. faced_cb  -> Have you ever faced Cyberbullying or cyber harassment, or blackmailing?
# 06. shared_personal  -> Have you ever mistakenly shared your personal information on any Phishing site or any other suspicious sites?
# 07. have_knowledge  -> Do you think females have enough knowledge about protecting their personal privacy?
# 08. more_aware  -> Do you think females need to be more aware of cyberbullying?
# 09. have_law  -> Is our Bangladeshi law sufficient to prevent cyberbullying?
# 10. ask_help  -> To whom will you ask for help to protect yourself if you face cyberbullying?
# 11. harm_reporting  -> Do you think that cyberbullying is a personal problem so reporting that problem and making them public can harm the dignity of females?
# 12. how_protect  -> How do you think we can protect females from cyberbullying?
# 

# In[116]:


# import library..
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import matplotlib.pyplot as plt
import seaborn
get_ipython().run_line_magic('matplotlib', 'inline')
import sklearn.cluster as cluster
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
import graphviz
# import the library
from sklearn.model_selection import train_test_split 
from IPython.display import Image  
import pydotplus
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import  GaussianNB


# In[117]:


myFile=pd.read_csv('cyber_bullying_final2.csv')
myFile


# In[118]:


print(myFile.shape)


# In[119]:


help_protect_column= myFile.iloc[:,10:12]


# In[120]:


help_protect_column


# In[121]:


ask_help = myFile.iloc[:,10]
ask_help


# In[122]:


how_protect = myFile.iloc[:,11]
how_protect


# In[123]:


seperate_label = myFile.iloc[:,:-2]


# In[124]:


seperate_label


# In[125]:


seperate_label.idea_about_cb.value_counts()


# In[126]:


convert_lebel= seperate_label.apply(LabelEncoder().fit_transform)


# In[127]:


convert_lebel


# In[128]:


convert_lebel.gender.value_counts()


# In[129]:


convert_lebel.idea_about_cb.value_counts()


# In[130]:


convert_lebel.face_cb.value_counts()


# In[131]:


convert_lebel.insert(10,'ask_help',ask_help)


# In[132]:


convert_lebel


# In[133]:


convert_lebel.ask_help.value_counts()


# In[134]:


convert_lebel.more_aware.value_counts()


# In[135]:


convert_lebel.insert(11,'how_protect',how_protect)


# In[136]:


convert_lebel


# In[137]:


female_lebel = convert_lebel[convert_lebel['gender'] == 0]


# In[138]:


female_lebel


# In[139]:


print(female_lebel.shape)


# In[140]:


female_lebel.more_aware.value_counts()


# In[141]:


female_lebel.idea_about_cb.value_counts()


# In[142]:


seaborn.countplot(x='idea_about_cb',data=female_lebel,palette='winter',edgecolor= seaborn.color_palette('dark',n_colors=1))
seaborn.set(rc={'figure.figsize':(5.7,3.5)})


# In[143]:


female_lebel.face_cb.value_counts()


# In[144]:


seaborn.countplot(x='idea_about_cb',hue='face_cb',data=female_lebel,palette='colorblind',edgecolor= seaborn.color_palette('dark',n_colors=1))
seaborn.set(rc={'figure.figsize':(5.7,3.5)})


# In[145]:


seaborn.countplot(x='age',hue='face_cb',data=female_lebel,palette='winter',edgecolor= seaborn.color_palette('dark',n_colors=1))
seaborn.set(rc={'figure.figsize':(5.7,3.5)})


# In[146]:


seaborn.countplot(x='face_cb',hue='ask_help',data=female_lebel,palette='colorblind',edgecolor= seaborn.color_palette('dark',n_colors=1))
seaborn.set(rc={'figure.figsize':(15.7,6.4)})


# In[147]:


seaborn.countplot(x='face_cb',hue='how_protect',data=female_lebel,palette='colorblind',edgecolor= seaborn.color_palette('dark',n_colors=1))
seaborn.set(rc={'figure.figsize':(7.7,3.5)})


# # Machine Learning algorithm apply start.......

# In[148]:


female_lebel


# # Seperate input and output column

# In[149]:


y= female_lebel['more_aware']
y.head()


# In[150]:


x= female_lebel
x.drop('more_aware', inplace=True, axis=1)


# In[151]:


x


# # Seperate training and testing dataset

# In[152]:


xtrain, xtest, ytrain, ytest= train_test_split(x,y, test_size= 0.30, random_state=1)


# # Decission Tree

# In[153]:


dt_model= DecisionTreeClassifier()


# In[154]:


dt_model.fit(xtrain,ytrain)


# In[155]:


print("Decission Tree Result: ")
print(f'Training Accuracy - : {dt_model.score(xtrain,ytrain)*100:.3f} %')
print(f'Testing Accuracy - : {dt_model.score(xtest,ytest)*100:.3f} %')


# # Confusion matrix for decission tree

# In[156]:


plot_confusion_matrix(dt_model, xtest, ytest)  
plt.show()


# In[157]:


from sklearn.metrics import classification_report
print(classification_report(ytest, dt_model.predict(xtest)))


# # K- nearest neighbors Algorithm

# In[158]:


knn= KNeighborsClassifier(n_neighbors=22)


# In[159]:


knn.fit(xtrain,ytrain)


# In[160]:


from sklearn.metrics import accuracy_score
print("K- nearest neighbors Result: ")
print(f"Training Accuracy Score: {accuracy_score(ytrain, knn.predict(xtrain)) * 100:.3f}%")
print(f"Testing Accuracy Score: {accuracy_score(ytest, knn.predict(xtest)) * 100:.3f}%")


# Confusion matrix for K- nearest neighbors Algorithm

# In[161]:


plot_confusion_matrix(knn, xtest, ytest)  
plt.show()


# In[162]:


print(classification_report(ytest, knn.predict(xtest)))


# # Naive Bayes

# In[163]:



gnv= GaussianNB()


# In[164]:


gnv.fit(xtrain,ytrain)


# In[165]:


print("Naive Bayes Result: ")
print(f"Training Accuracy Score: {gnv.score(xtrain, ytrain) * 100:.3f}%")
print(f"Testing Accuracy Score: {gnv.score(xtest,ytest) * 100:.3f}%")


# Confusion matrix for Naive Bayes

# In[166]:


plot_confusion_matrix(gnv, xtest, ytest)  
plt.show()


# In[167]:


print(classification_report(ytest, gnv.predict(xtest)))


# # Random Forest

# In[168]:


# import library
from sklearn.ensemble import RandomForestRegressor
rf_model= RandomForestRegressor(n_estimators=100,max_depth=8)
from sklearn.metrics import mean_absolute_error, r2_score


# In[169]:


rf_model.fit(xtrain,ytrain)


# In[170]:


print("Random Forest Result: ")
print(f'Traning Accuracy - : {rf_model.score(xtrain,ytrain)*100:.3f} %')
print(f'Testing Accuracy - : {rf_model.score(xtest,ytest)*100:.3f} %')


# Confusion matrix for Random Forest

# In[ ]:




