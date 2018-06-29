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
from sklearn import neighbors

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

train_feats, test_feats, train_labels, test_labels = tts(x_data, y_data, test_size=0.1, random_state=42)


# machine_learning = neighbors.KNeighborsRegressor(n_neighbors=5)
# machine_learning = tree.DecisionTreeRegressor(max_depth=12)
machine_learning = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=2, loss='ls')

# train
machine_learning.fit(train_feats, train_labels)

# prediction
predictions = machine_learning.predict(test_feats)
print("Predictions:")
print(predictions)
print("\n")


dataUser = data.copy()
dataUser = dataUser.drop('userID', 1)
dataUser = dataUser.drop('rating', 1)
dataUser = dataUser.drop_duplicates()
dataUser.insert(0, 'userID', 170)
previsao = machine_learning.predict(dataUser)
i = 0
top_filmes = list()
for index, row in dataUser.iterrows():
    top_filmes.append((int(row["movieID"]), previsao[i]))
    i += 1
top_filmes.sort(key=lambda x: x[1])  # ordena de forma crescente por rating
top_filmes.reverse()  # reverse a ordenação, fazendo com que os maiores rating fiquem no início
top_cinco = pd.DataFrame.from_records(top_filmes[:5], columns=["movieID", "prediction"])

nomes = pd.read_csv(os.sep.join(['movies.csv']), sep=';')

resultado = pd.merge(top_cinco, nomes, on="movieID")[["movieID", "title", "prediction"]]


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
