import os, sys
import datetime

from matplotlib.colors import ListedColormap
from pandas import DataFrame

import pandas as pd
import numpy as np
import scipy as sci

from scipy.spatial import distance

import matplotlib.pyplot as plt
import pylab as pl

from sklearn import neighbors
from sklearn.model_selection import cross_val_predict


class DataModelling:
    data = pd.DataFrame()

    def __init__(self):
        pass

    def read_data(self, input_path):
        print(self.data.dtypes)
        self.data = pd.read_csv(input_path)
        print("original data...")
        print(self.data)
        self.data['TimeSt'] = self.data['TimeSt'].astype('datetime64[ns]')

    def clean(self):
        # clean data of rows with duplicated dates and geoInfo
        # original dataset has space on column name that is fixed on input daatset file
        self.data = self.data.drop_duplicates(['TimeSt', 'Latitude', 'Longitude'])
        print ("cleaned data...")
        print(self.data)
        # plt.ion()
        # plt.plot(self.data['Latitude'], self.data['Longitude'])
        # plt.title("Raw input")

    def label_and_model(self):
        class_path = os.path.join(os.getcwd(), 'data/POIList.csv')
        lables_data_class = pd.read_csv(class_path)['POIID']
        features_data = np.array(self.data[['Latitude', 'Longitude']])
        lables_data_train = np.array(pd.read_csv(class_path)[['Latitude', 'Longitude']])

        # convert nominal classes to numeric
        lables_data_class = np.array(lables_data_class.replace({'POI1': 1, 'POI2': 2, 'POI3': 3, 'POI4': 4}))

        # phase1: training
        #
        knn_clf = neighbors.KNeighborsClassifier(n_neighbors=3,  weights='distance')
        knn_clf.fit(lables_data_train, lables_data_class)

        # phase2: testing
        # Model based on KNN
        features_data_class = knn_clf.predict(features_data)

        # Average
        avg_lat, avg_long = np.average(features_data, 0), np.average(features_data, 1)
        print(avg_lat, avg_long)
        # Standard Deviation
        sd_lat, sd_long = np.std(features_data, 0), np.std(features_data, 1)
        print(sd_lat, sd_long)

        # classifier grid

        x_min, x_max = features_data[:, 0].min(), features_data[:, 0].max()
        y_min, y_max = features_data[:, 1].min(), features_data[:, 1].max()
        h = 0.2
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        Z = knn_clf.predict(np.c_[xx.ravel(), yy.ravel()])

        Z = Z.reshape(xx.shape)
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#ddffaa'])
        pl.figure(1, figsize=(10, 10))
        # predicted data
        pl.pcolormesh(xx, yy, Z, cmap=cmap_light)

        # classes centroids
        pl.scatter(features_data[:, 0], features_data[:, 1], c=features_data_class, cmap=cmap_light)
        pl.xlabel('Latitude')
        pl.ylabel('Longitude')
        pl.title('Classes')
        pl.gcf().canvas.set_window_title('POI classified')
        pl.show()


if __name__ == '__main__':
    print("start reading data....")
    input_path = os.path.join(os.getcwd(), 'data/DataSample.csv')
    print(input_path)
    datamodelling = DataModelling()
    datamodelling.read_data(input_path)
    datamodelling.clean()
    datamodelling.label_and_model()
