from sklearn import datasets
from sklearn import neighbors
from sklearn import metrics
from sklearn.model_selection import train_test_split as tts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

boston = datasets.load_boston()

'''bos = pd.DataFrame(boston['data'])
bos.columns = boston['feature_names']
bos['Price'] = boston['target']
print(bos)'''

train_feats, test_feats, train_labels, test_labels = tts(boston['data'], boston['target'], test_size=0.3)

KNR = neighbors.KNeighborsRegressor(n_neighbors=8)

'''train_feats = boston['data'][:500]
train_labels = boston['target'][:500]'''

#train
KNR.fit(train_feats, train_labels)

#prediction
'''test_feats = np.array(boston['data'][504])
test_feats = test_feats.reshape(1,-1)
print(bos.ix[500:])'''

predictions = KNR.predict(test_feats)
'''print(predictions)
print("\n")'''

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
