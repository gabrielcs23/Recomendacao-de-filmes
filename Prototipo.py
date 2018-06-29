from sklearn import metrics
from sklearn.model_selection import train_test_split as tts
from sklearn import linear_model
import pandas as pd
import numpy as np
import os
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


filepath = os.sep.join(['movie_dataV18col.csv'])
data = pd.read_csv(filepath, sep=';')
data.replace('N/A', np.NaN)
data = data.apply(lambda x: x.fillna(x.mean()), axis=0)

y_col = 'rating'

feature_cols = [x for x in data.columns if (x != y_col)]
x_data = data[feature_cols]
y_data = data[y_col]

train_feats, test_feats, train_labels, test_labels = tts(x_data, y_data, test_size=0.1)


# KNR = neighbors.KNeighborsRegressor(n_neighbors=5)
# KNR = linear_model.LassoCV(cv=20)
KNR = linear_model.LinearRegression()
# KNR = svm.SVR(kernel='poly', C=1e3, degree=2)
# KNR = linear_model.SGDRegressor(loss='epsilon_insensitive',penalty='elasticnet',epsilon=0.1)
# KNR = tree.DecisionTreeRegressor(max_depth=12)
# KNR = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,max_depth=2, loss='ls')
# KNR = linear_model.Lars()
# KNR = linear_model.BayesianRidge()
# KNR = linear_model.LogisticRegression(class_weight='balanced', solver='saga')
# KNR = linear_model.LassoLars()
# KNR = linear_model.ARDRegression()

'''train_feats = boston['data'][:500]
train_labels = boston['target'][:500]'''

# train
KNR.fit(train_feats, train_labels)

# prediction
predictions = KNR.predict(test_feats)
print("Predictions:")
print(predictions)
print("\n")


dataUser = data.copy()
dataUser = dataUser.drop('userID', 1)
dataUser = dataUser.drop('rating', 1)
dataUser = dataUser.drop_duplicates()
dataUser.insert(0, 'userID', 170)
eai = KNR.predict(dataUser)
i = 0
top_filmes = list()
for index, row in dataUser.iterrows():
    top_filmes.append((int(row["movieID"]), eai[i]))
    i += 1
top_filmes.sort(key=lambda x: x[1])  # ordena de forma crescente por rating
top_filmes.reverse()  # reverse a ordenação, fazendo com que os maiores rating fiquem no início
top_cinco = pd.DataFrame.from_records(top_filmes[:6], columns=["movieID", "prediction"])

nomes = pd.read_csv(os.sep.join(['movies.csv']), sep=';')

resultado = pd.merge(top_cinco, nomes, on="movieID")[["movieID", "title"]]


print("Result:\n", resultado, "\n")

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
