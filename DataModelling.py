import os, sys
import datetime

from matplotlib.colors import ListedColormap
from pandas import DataFrame
from matplotlib.backends.backend_pdf import PdfPages

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
        # data in them input data from DataSample.csv
        self.data = pd.read_csv(input_path)
        self.data['TimeSt'] = self.data['TimeSt'].astype('datetime64[ns]')
        print("original data...")
        # print(self.data)

    def clean(self):
        # clean data of rows with duplicated dates and geoInfo
        # original dataset has space on column name that is fixed on input dataset file
        self.data = self.data.drop_duplicates(['TimeSt', 'Latitude', 'Longitude'])
        print ("cleaned data...")
        # print(self.data)
        # plt.ion()
        # plt.plot(self.data['Latitude'], self.data['Longitude'])
        # plt.title("Raw input")

    def distance(self, nodes):
        #    calculate euclidean distance given 2 points
        for node in nodes:
            print(node, "+++++")
            print (type(node))
            euclidean_dist = np.sqrt(np.sum(np.power(node[0][0] - node[1][0]), np.power(node[0][1] - node[1][1])))
        euclidean_dist

    def min_dist_label_and_model(self):
        features_data = np.array(self.data[['Latitude', 'Longitude', 'Country', 'Province', 'City']])
        # print features_data
        for d in features_data:
            distance(d)
        pass

    def knn_label_and_model(self):

        # labelling by classes in POIList
        class_path = os.path.join(os.getcwd(), 'data/POIList.csv')
        lables_data_class = pd.read_csv(class_path)

        # convert nominal classes to numeric
        data_class = np.array(lables_data_class['POIID'].replace({'POI1': 1, 'POI2': 2, 'POI3': 3, 'POI4': 4}))
        print(type(lables_data_class))

        features_data = np.array(self.data[['Latitude', 'Longitude']])

        print ("features_data type: ")
        # print (features_data)
        print (type(features_data))

        data_train = np.array(lables_data_class[['Latitude', 'Longitude']])

        # phase1: classifier and training
        knn_clf = neighbors.KNeighborsClassifier(n_neighbors=4, weights='distance')
        knn_clf.fit(data_train, data_class)

        # phase2: testing - Model based on KNN
        features_data_class = knn_clf.predict(features_data)
        print("knn predict result:")
        print(features_data_class)

        # Average / mean
        print("Calculate mean...")
        mean_lat, mean_long = np.mean(features_data, axis=0)[0], np.mean(features_data, axis=0)[1]
        print(mean_lat)
        print(mean_long)

        # Standard Deviation
        print("Calculate Standard deviation...")
        # axis=0 is for each column
        sd_lat, sd_long = np.std(features_data, axis=0)[0], np.std(features_data, axis=0)[1]
        print(sd_lat)
        print(sd_long)

        # classifier grid
        # get grid dimensions (min/max X/Y)
        x_min, x_max = features_data[:, 0].min(), features_data[:, 0].max()
        y_min, y_max = features_data[:, 1].min(), features_data[:, 1].max()
        h = 0.2
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Z = knn_clf.predict(np.c_[xx.ravel(), yy.ravel()])
        # Z = Z.reshape(xx.shape)

        # FFAAAA -> pink
        # AAFFAA -> green
        # AAAAFF -> purple
        # ffc8aa -> orange
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#ffc8aa'])

        pl.figure(1, figsize=(10, 10))

        # pl.pcolormesh(xx, yy, Z, cmap=cmap_light)

        # classes centroids
        pl.scatter(features_data[:, 0], features_data[:, 1], c=features_data_class, cmap=cmap_light)
        # [ features_data[row]['Class'] = features_data_class[row] for row in features_data ]

        # pl.scatter(features_data_class[0], features_data_class[1], c=features_data_class, cmap=cmap_light)
        plt.colorbar()
        pl.xlabel('Latitude')
        pl.ylabel('Longitude')
        pl.title('Classified POI based on Long/Lat')
        pl.gcf().canvas.set_window_title('Classified POI')

        output = os.path.join(os.getcwd(), 'output')
        if not os.path.exists(output):
            os.mkdir(os.path.join(output))
        # pdf = PdfPages(os.path.join(output, 'results'))
        pl.plot(features_data_class.all())
        pl.savefig(output)
        pl.show(features_data_class.all())

        # plt.plot(features_data[:, 0], features_data[:, 1], c=features_data_class, cmap=cmap_light)


if __name__ == '__main__':
    print("start reading data....")
    input_path = os.path.join(os.getcwd(), 'data/DataSample.csv')
    print(input_path)

    # get algo method
    op = sys.argv[1]
    datamodelling = DataModelling()
    datamodelling.read_data(input_path)
    datamodelling.clean()
    algo_options = {
        "knn":
            datamodelling.knn_label_and_model,
        "min_dist":
            datamodelling.min_dist_label_and_model
    }
    algo_options[op]()
