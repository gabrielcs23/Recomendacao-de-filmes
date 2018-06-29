from sklearn import datasets
from sklearn import neighbors
from sklearn import metrics
from sklearn.model_selection import train_test_split as tts
from sklearn import linear_model
from sklearn import kernel_ridge
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import csv
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.ensemble import GradientBoostingRegressor

def round_of_rating(number):
    """Round a number to the closest half integer.
    >>> round_of_rating(1.3)
    1.5
    >>> round_of_rating(2.6)
    2.5
    >>> round_of_rating(3.0)
    3.0
    >>> round_of_rating(4.1)
    4.0"""

    return round(number * 2) / 2

#boston = datasets.load_boston()

'''bos = pd.DataFrame(boston['data'])
bos.columns = boston['feature_names']
bos['Price'] = boston['target']
print(bos)'''

filepath = os.sep.join(['movie_data.csv'])
data = pd.read_csv(filepath, sep=';')
data.replace('N/A', np.NaN)
data = data.apply(lambda x: x.fillna(x.mean()),axis=0)

y_col = 'rating'

feature_cols = [x for x in data.columns if (x!=y_col)]
x_data = data[feature_cols]
y_data = data[y_col]

train_feats, test_feats, train_labels, test_labels = tts(x_data, y_data, test_size=0.1)

'''scaler = StandardScaler()
scaler.fit(train_feats)  # Don't cheat - fit only on training data
train_feats = scaler.transform(train_feats)
test_feats = scaler.transform(test_feats)'''

#KNR = neighbors.KNeighborsRegressor(n_neighbors=5)
#KNR = linear_model.LassoCV(cv=20)
KNR = linear_model.LinearRegression()
#KNR = svm.SVR(kernel='poly', C=1e3, degree=2)
#KNR = linear_model.SGDRegressor(loss='epsilon_insensitive',penalty='elasticnet',epsilon=0.1)
#KNR = tree.DecisionTreeRegressor(max_depth=12)
#KNR = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,max_depth=2, loss='ls')
#KNR = linear_model.Lars()
#KNR = linear_model.BayesianRidge()
#KNR = linear_model.LogisticRegression(class_weight='balanced', solver='saga')
#KNR = linear_model.LassoLars()
#KNR = linear_model.ARDRegression()

'''train_feats = boston['data'][:500]
train_labels = boston['target'][:500]'''

#train
KNR.fit(train_feats, train_labels)

#prediction
'''test_feats = np.array(boston['data'][504])
test_feats = test_feats.reshape(1,-1)
print(bos.ix[500:])'''
# teste = [x for x in test_feats]
# print(test_feats) # movieID do prdetiction cuja rating > 4
predictions = KNR.predict(test_feats)
print("Predictions:")
print(predictions)
print("\n")
'''print(test_feats['movieID'][1])
print("\n")'''

score = 0
i = 0
for x in test_labels:
    if(i>-1 and x==round_of_rating(predictions[i])):
        score += 1
    i+=1
i = 0
result = []
user = 170 # usuário para que a recomendação está sendo feita
for index, row in test_feats.iterrows():
    if row["userID"] == user:
        result.append([row["movieID"], predictions[i], row["userID"]])
    i += 1
result.sort(key=lambda x: x[1]) # ordena de forma crescente por rating
result.reverse() # reverse a ordenação, fazendo com que os maiores rating fiquem no início

print("Result = ", result[:5])
score = score/len(predictions)
print("Average score:")
print(score)
print("\n")

print("Explained variance score:")
print(metrics.explained_variance_score(test_labels, predictions))
print("\n")
print("Mean absolute error:")
print(metrics.mean_absolute_error(test_labels, predictions))
print("\n")
print("Mean squared error:")
print(metrics.mean_squared_error(test_labels, predictions))
print("\n")
print("Mean squared log error:")
print(metrics.mean_squared_log_error(test_labels, predictions))
print("\n")
print("Median absolute error:")
print(metrics.median_absolute_error(test_labels, predictions))
print("\n")
print("R2 score:")
print(metrics.r2_score(test_labels, predictions))
