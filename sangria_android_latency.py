"""
quick test sklearn
"""

print("Capturing Model Latencies on Android")
print("-- Danish Gufran ")

import os
import numpy as np
import sklearn
from time import time
from sklearn import neighbors
from sklearn.svm import SVC as SVM
# import keras
# from keras.saving import pickle_utils
import pickle

# create dataset
# (5472, 30, 30, 1) (5472, 1) (1368, 30, 30, 1) (1368, 1)
# generate random numbers with shape given and in range 0 to 100
x_train = np.random.randint(100, size=(5472, 900))
y_train = np.random.randint(382, size=(5472))

x_test = np.random.randint(100, size=(1000, 1, 165))
#y_test = np.random.randint(382, size=(1000))

################################################################
# KNN
################################################################
# print("Training KNN model.")
# knn = neighbors.KNeighborsRegressor(n_neighbors=3)
# knn.fit(x_train, y_train)
# pickle.dump(knn, open("models/knn.pickle", 'wb'))


# os.chdir('/Users/danishgufran/Desktop/papers/ESL_SANGRIA/Supp/')
print(os.getcwd())
# dev = ['BLU','HTC','LG','MOTO','OP3','S7']

dev = ['KNN', 'SVM', 'RF', 'GPC']

store = []
for device in dev:

    x_test = np.random.randint(100, size=(1000, 1, 165))
    if device == 'CNN':
      x_test = np.random.randint(100, size=(1000, 1, 1,165))

    knn_file = 'android_pickle/' + str(device) + "_OP3.pkl"
    print(knn_file)
    knn = pickle.load(open(knn_file, 'rb'))

    print(f"{device} warm up")
    for row in x_test[:]:
        y_pred = knn.predict(row)

    print(f"Testing {device}")
    start_time = time()
    for row in x_test:
        y_pred = knn.predict(row)

    passed_time = time() - start_time
    fps = passed_time*1000/x_test.shape[0]
    store.append(fps)

    print(f"{device} Latency: {fps:.2f} ms")
print(store)

    ################################################################
    # SVM
    ################################################################
    # print("Training SVM model.")
    # svm = SVM(C=1000)
    # svm.fit(x_train, y_train)
    # pickle.dump(svm, open("models/svm.pickle", 'wb'))


    # svm = pickle.load(open("models/svm.pickle", 'rb'))

    # print("SVM warm up")
    # for row in x_test[:100]:
    #     y_pred = svm.predict(row)


    # print("Testing SVM")
    # start_time = time()
    # for row in x_test:
    #     y_pred = svm.predict(row)

    # passed_time = time() - start_time
    # fps = passed_time*1000/x_test.shape[0]

    # print(f"SVM Latency: {fps:.2f} ms")

    ################################################################