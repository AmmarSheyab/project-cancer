# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 16:49:52 2022

@author: ASUS
"""

import numpy as np
import pandas as pd

df=pd.read_csv('cancer.csv')

df.info()
a=df.isnull().sum()

X=df.loc[:,df.columns.difference(['diagnosis'],sort=False)].values
y=df.iloc[:,1]

from sklearn.preprocessing import LabelEncoder
labelEncoder=LabelEncoder()
y=labelEncoder.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=(42))

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 42)
classifier.fit(X_train, y_train)

#Predicting the Test set results
y_pred = classifier.predict(X_test)

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#print(cm)


'''
70+41+1+2=114
accurcy =(70+41)/114
'''
print('---------accurcy--------',(70+41)/114)
'''
70+41+1+2
error =(1+2)/114
'''
print('---------error----------',(1+2)/114)


import statsmodels.api as sm
X=np.append(np.ones((len(X),1)).astype(int),values=X,axis=1)#CONSTANT
#import statsmodels.api as sm


#A function to determine the veggies that have an effect on the results
def reg_ols(X,y):
    columns=list(range(X.shape[1]))
    
    for i in range(X.shape[1]):
        X_opt=np.array(X[:,columns],dtype=float) 
        regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
        pvalues = list(regressor_ols.pvalues)
        d=max(pvalues)
        if (d>0.05):
            for k in range(len(pvalues)):
                if(pvalues[k] == d):
                    del(columns[k])  
    
    return(X_opt,regressor_ols)

X_opt,regressor_ols=reg_ols(X, y)
regressor_ols.summary()


from sklearn.model_selection import train_test_split
X_train_opt,X_test_opt,y_train_opt,y_test_opt=train_test_split(X_opt,y,test_size=0.2,random_state=(0))
'''
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_opt = sc.fit_transform(X_train_opt)
X_test_opt = sc.transform(X_test_opt)
'''
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train_opt, y_train_opt)

#Predicting the Test set results
y_pred1 = classifier.predict(X_test_opt)

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cc = confusion_matrix(y_test_opt, y_pred1)
#print(cc)


'''
63+44+4+3=114
accurcy opt =(63+44)/114
'''
print('---------accurcy_opt ------',(63+44)/114)
'''
63+44+4+3=114
error opt =(4+3)/114
'''
print('--------------error_opt----------',(4+3)/114)



