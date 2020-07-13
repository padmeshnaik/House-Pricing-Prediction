import pandas as pd
import numpy as np
import collections
import seaborn as sns
import matplotlib.pyplot as  plt
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

from sklearn.preprocessing import StandardScaler


import statsmodels.api as sm

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from sklearn.preprocessing import PolynomialFeatures


from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

import pickle




df=pd.read_csv("C:/Users/Padmesh/Desktop/Machine Learning Models/Regression Models/House_Prices/House_Data.csv")
df['price']=np.log(df['price'])

df['sqft_living']=np.log(df['sqft_living'])

df["sqft_basement"]=np.log1p(df['sqft_basement'])



#print(df.info())

X_old = df.iloc[:,:21]


y = df.iloc[:,[2]].values

X=X_old.drop(df.columns[[1,2]],axis=1)

#print(df.info())



"""

def adjustedR2(r2,n,k):
    return 1-[(n-1)/(n-k-1)]*(1-r2)


# Computing the correlation matrix for all features with the 'Price' parameter.

#cols = ['price', 'lat', 'long', 'sqft_living15', 'sqft_lot15']
cols = ['price','sqft_living', 'sqft_above','bathrooms']
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm,
                 cbar=True,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 15},
                 yticklabels=cols,
                 xticklabels=cols)

plt.tight_layout()
# plt.savefig('images/10_04.png', dpi=300)
plt.show()

# calculate Pearson's correlation
corr, _ = pearsonr(df['sqft_living'], df['sqft_above'])
print('Pearsons correlation: %.3f' % corr)
"""

"""
Here we notice the correlation between the features 'sqft_living' and 'sqft_above' is very high.
So to prevent our model from overfitiing, we remove one of them.
"""
"""

# Drawing a regression line between 'Price' and 'sqft_living'.
# Since 'sqft_living' has the highly correlated with our target column('Price').

X = df.iloc[:, 5].values.reshape(-1, 1)  # values converts it into a numpy array
Y = df.iloc[:, 2].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, Y)  # perform linear regression
Y_pred = linear_regressor.predict(X)  # make predictions

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.xlabel('Prices')
plt.ylabel('sqft_living')
plt.show()

"""
"""
We have chance to plot most of them and reach some useful analytical results.
Drawing charts and examining the data before applying a model is a very good practice because we 
may detect some possible outliers or decide to do normalization. This is not a must but get know the 
data is always good. Then, I started with the histograms of dataframe.
"""



#####  To count the occurences of each values in a particular column

"""
views=df['view']

print(collections.Counter(views))  # return key value pair where value refers the the count.

#print(np.bincount(views))
"""
"""

cols = ['price','view','grade', 'sqft_basement','lat']
sns.pairplot(df[cols], height=2)
plt.tight_layout()
# plt.savefig('images/10_03.png', dpi=300)
#plt.show()

"""

# We plot histogram for each features to check whether any outliers are present in that column.


"""

plt.hist(df['bedrooms'])
plt.xlabel('bedrooms')
plt.ylabel('count')
plt.tight_layout()
plt.show()


# You can also draw boxplot to identify the outliers.

plt.boxplot(df['bedrooms'])
plt.xlabel('bedrooms')
plt.ylabel('count')
plt.tight_layout()
plt.show()
"""

"""
X = df.iloc[:,[5,11]].values
y = df.iloc[:, 2].values


#X=X.reshape(-1,1)   # Reshaping the 1D array to a 2D array.(Only when the training part has only 1 column)

print(X)


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

model = LinearRegression()
model.fit(X_train,Y_train)
print('Model score: '+str(model.score(X_test,Y_test)))
"""

"""
#### We use SelectKBest library to select the best features based on their correlation with target column

best_features=SelectKBest(score_func=f_regression,k=7)
fit=best_features.fit(X,y)

df_scores=pd.DataFrame(fit.scores_)   # Will  create a new column having all the scores
df_columns=pd.DataFrame(X.columns)

feature_scores=pd.concat([df_columns,df_scores],axis=1)
feature_scores.columns=['Features','Scores']

print(feature_scores.nlargest(10,'Scores'))

"""
"""
#### Backward Feature Elimination

## Performing OLS to find out the important features

X = np.append(arr = np.ones((X.shape[0],1)).astype(int), values = X, axis = 1)

#print(X[:,[0]])


X_opt = X[:,[0,4,10,11,18,3,8,12,2,16,7]]


regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()   
print(regressor_OLS.summary())

# In the above summary we find out that there is a strong multicollinearity problem present in our
# training dataset which is indicated by the high condition no.
# So we remove one of the two highly correlated features 'sqft_living' and 'sqft_above'.
# P.S Removing highly correlated features is not recommended to be done everytime.


X = np.append(arr = np.ones((X.shape[0],1)).astype(int), values = X, axis = 1)
X_opt = X[:,[0,4,10,3,8,12,16,7]]
#X_opt = X[:,0:20]


regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()   
print(regressor_OLS.summary())

"""
"""
X_train, X_test, Y_train, Y_test = train_test_split(X.iloc[:,[2,3,6,7,9,11,15]].values, y, test_size = 0.2, random_state = 0)

#X_train, X_test, Y_train, Y_test = train_test_split(X.iloc[:,2:19].values, y, test_size = 0.2, random_state = 0)



model = LinearRegression()
model.fit(X_train,Y_train)
print('Model score: '+str(model.score(X_test,Y_test)))
print("Training score for model:"+str(model.score(X_train,Y_train)))


### It is mandatory to normalize the data before giving it to any regularization function.
# So we need to keep 'normalize' as true inside both the regularization functions.




rr=Ridge(alpha=0.1,normalize=True)
rr.fit(X_train,Y_train)
print('Regularization Ridge Model score: '+str(rr.score(X_test,Y_test)))
print("Training score for ridge:"+str(rr.score(X_train,Y_train)))

# When we train the model with all the data , the accuracy we achieve is 69.5%
# Wheras when we train the model with just 7 features,the accuracy obtained is 64.7%.
# So we move forward with the second model as it has less number of features(less computation).

lasso = Lasso(alpha=100, max_iter=10000,normalize=True)            # Default value for alpha is 1
lasso.fit(X_train,Y_train)
print('Regularization Lasso Model score: '+str(lasso.score(X_test,Y_test)))
print("Training score for lasso:"+str(lasso.score(X_train,Y_train)))

"""

####  Applying Polynomial Regression

## To overcome under-fitting, we need to increase the complexity of the model.
## To convert the original features into their higher order terms we will use
##  the PolynomialFeatures class provided by scikit-learn. 



polynomial_features= PolynomialFeatures(degree=2)

X_main=X.iloc[:,[2,3,7,9,11,15]].values


x_poly = polynomial_features.fit_transform(X_main)

X_train, X_test, Y_train, Y_test = train_test_split(x_poly, y, test_size = 0.2, random_state = 0)


rr1=Ridge(alpha=0.001)
rr1.fit(X_train,Y_train)
print('Regularization Ridge Polynomial Model Training score: '+str(rr1.score(X_train,Y_train)))
print('Regularization Ridge Polynomial Model Testing score: '+str(rr1.score(X_test,Y_test)))

"""

#### Backward Feature Elimination

## Performing OLS to find out the important features


X_main=X.iloc[:,[2,3,7,9,11,15]].values
X_main1 = np.append(arr = np.ones((X.shape[0],1)).astype(int), values = X_main, axis = 1)
X_opt = X_main1



regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()   
print(regressor_OLS.summary())

"""

"""

lasso1 = Lasso(alpha=0.001, max_iter=10000)            # Default value for alpha is 1
lasso1.fit(X_train,Y_train)
print('Regularization Lasso Polynomial Model Training score: '+str(lasso1.score(X_train,Y_train)))
print('Regularization Lasso Polynomial Model Testing score: '+str(lasso1.score(X_test,Y_test)))

"""

### We conclude that , the ridge regularization function is able to penaltize the coefficients without
### causing any problem like overfitting and underfitting(It does but not to a great extent)

## So we choose ridge over lasso.


### K-fold cross-validation

"""


kfold = StratifiedKFold(n_splits=10,
                        random_state=1,shuffle=False).split(X_train, Y_train)


scores = []
for k, (train, test) in enumerate(kfold):          # kfold will have the subsets of train and test data.
    rr1.fit(X_train[train], Y_train[train])
    score = rr1.score(X_train[test], Y_train[test])
    scores.append(score)
    print('Fold: %2d, Acc: %.3f' % (k+1,score))
    
print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


"""

# Scikit-learn also implements a k-fold cross-validation scorer, which
# allows us to evaluate our model using stratified k-fold cross-validation

"""
scores = cross_val_score(estimator=rr1,
                         X=X_train,
                         y=Y_train,
                         cv=10,
                         n_jobs=-1)
print('CV accuracy scores: %s' % scores)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


"""
#### So after finish the modeling process and doing the analysis of our model.
#### We come up with the following conclusion.
#### When we train our model with all the data(17 features), we obtain an accuracy 0f 81%.
#### When we train our model with only 7 important features, we obtain an accuracy of 72.5%.


##### Creating a flask we application which predicts the house prices.

pickle.dump(rr1, open('C:/Users/Padmesh/Desktop/Machine Learning Models/Regression Models/House_Prices/model.pkl','wb'))

