# import libraries
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import seaborn as sns

import warnings
warnings.simplefilter(action='ignore', category=Warning)


################################
########## DATA PREP ###########
################################

# Load in the data
df = pd.read_csv("predicting_num_sold.csv")

# Split into train and test sections
y = df.pop("num_sold")
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)


# target encoding
from sklearn.base import BaseEstimator, TransformerMixin

class TargetEncode(BaseEstimator, TransformerMixin):
    """
    categories: The column names of the features you want to target-encode
    k(int): min number of samples to take a category average into the account
    f(int): Smoothing effect to balance the category average versus the prior probability, or the mean value relative to all the training  
            examples
    noise_level: The amount of noise you want to add to the target encoding in order to avoid overfitting.
    random_state: The reproducibility seed in order to replicate the same target encoding while noise_level > 0
    """
    def __init__(self, categories='auto', k=1,f=1,noise_level=0,random_state=17):
        if type(categories) ==str and categories != 'auto':
            self.categories = [categories]
        else:
            self.categories = categories
        self.k = k
        self.f = f
        self.noise_level = noise_level
        self.encodings = dict()
        self.prior = None
        self.random_state = random_state
        
        
    def add_noise(self,series,noise_level):
        return series*(1+ noise_level*np.random.randn(len(series)))
    
    
    def fit(self, X,y=None):
        if type(self.categories) == 'auto':
            self.categories = np.where(X.dtypes == type(object()))[0]
            print(categories)
        temp = X.loc[:, self.categories].copy()
        temp['target'] = y
        self.prior = np.mean(y)
        for variable in self.categories:
            avg = (temp.groupby(by=variable)['target'].agg(['mean','count']))
            smoothing = (1/(1+np.exp(-(avg['count'] - self.k)/self.f)))
            self.encodings[variable] = dict(self.prior * 
                                              (1-smoothing) + avg['mean']*smoothing)
        return self
    
    
    def transform(self, X):
        Xt = X.copy()
        for variable in self.categories:
            Xt[variable].replace(self.encodings[variable], 
                                 inplace=True)
            unknown_value = {value:self.prior for value in 
                             X[variable].unique() if value 
                             not in self.encodings[variable].keys()}
            if len(unknown_value) > 0:
                Xt[variable].replace(unknown_value, inplace=True)
            Xt[variable] = Xt[variable].astype(float)
            if self.noise_level > 0:
                if self.random_state is not None:
                    np.random.seed(self.random_state)
                Xt[variable] = self.add_noise(Xt[variable],
                                             self.noise_level)
        return Xt
    
    
    def fit_transform(self,X,y=None):
        self.fit(X,y)
        return self.transform(X)
      
categories = ['country','store','product']
te = TargetEncode(categories = categories)
te.fit(X_train, y_train)
X_train = te.transform(X_train)
X_test = te.transform(X_test)
features = [col for col in X_test.columns if col not in ["row_id","date","dayofyear","week","month","day"]]
X_train = X_train[features]
X_test = X_test[features]
  

#################################
########## MODELLING ############
#################################

# Fit a model on the train section
params = {'n_estimators':1000, 
          'max_depth' : 7,
          'learning_rate': 0.1,
          'random_state':42}

regr = GradientBoostingRegressor(**params)
regr.fit(X_train, y_train)

# Report training set score
train_score = regr.score(X_train, y_train) * 100
# Report test set score
test_score = regr.score(X_test, y_test) * 100

# Write scores to a file
with open("metrics.txt", 'w') as outfile:
        outfile.write("Training variance explained: %2.1f%%\n" % train_score)
        outfile.write("Test variance explained: %2.1f%%\n" % test_score)



##########################################
##### PLOT FEATURE IMPORTANCE ############
##########################################
# Calculate feature importance in random forest
importances = regr.feature_importances_
labels = df.columns
feature_df = pd.DataFrame(list(zip(labels, importances)), columns = ["feature","importance"])
feature_df = feature_df.sort_values(by='importance', ascending=False,)

# image formatting
axis_fs = 18 #fontsize
title_fs = 22 #fontsize
sns.set(style="whitegrid")

ax = sns.barplot(x="importance", y="feature", data=feature_df)
ax.set_xlabel('Importance',fontsize = axis_fs) 
ax.set_ylabel('Feature', fontsize = axis_fs)#ylabel
ax.set_title('Random forest\nfeature importance', fontsize = title_fs)

plt.tight_layout()
plt.savefig("feature_importance.png",dpi=120) 
plt.close()


##########################################
############ PLOT RESIDUALS  #############
##########################################

y_pred = regr.predict(X_test) + np.random.normal(0,0.25,len(y_test))
y_jitter = y_test + np.random.normal(0,0.25,len(y_test))
res_df = pd.DataFrame(list(zip(y_jitter,y_pred)), columns = ["true","pred"])

ax = sns.scatterplot(x="true", y="pred",data=res_df)
ax.set_aspect('equal')
ax.set_xlabel('True wine quality',fontsize = axis_fs) 
ax.set_ylabel('Predicted wine quality', fontsize = axis_fs)#ylabel
ax.set_title('Residuals', fontsize = title_fs)

# Make it pretty- square aspect ratio
ax.plot([1, 10], [1, 10], 'black', linewidth=1)
plt.ylim((2.5,8.5))
plt.xlim((2.5,8.5))

plt.tight_layout()
plt.savefig("residuals.png",dpi=120) 
