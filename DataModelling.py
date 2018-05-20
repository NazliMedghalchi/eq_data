import os, sys
import datetime
from pandas import DataFrame

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

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
        plt.ion()
        plt.plot(self.data['Latitude'], self.data['Longitude'])
        plt.title("Raw input")
        plt.pause(10)

    def label_and_model(self):
        class_path = os.path.join(os.getcwd(), 'data/POIList.csv')
        lables_data_class = pd.read_csv(class_path)['POIID']
        lables_data_class = np.array(lables_data_class.replace({'POI1':1, 'POI2':2, 'POI3':3, 'POI4':4}))

        lables_data_train = np.array(pd.read_csv(class_path)[['Latitude', 'Longitude']])
        features_data = np.array(self.data[['Latitude', 'Longitude']])

        # phase1: training
        neightbours = neighbors.KNeighborsClassifier()
        neightbours.fit(lables_data_train, lables_data_class)

        # phase2: testing
        # Model based on KNN
        neightbours.predict(features_data)
        plt.ion()
        plt.plot(self.data['Latitude'], self.data['Longitude'])
        plt.title("Trained input")
        plt.pause(10)


    def analysis(self):
        pass


if __name__ == '__main__':
    print("start reading data....")
    input_path = os.path.join(os.getcwd(), 'data/DataSample.csv')
    print(input_path)
    datamodelling = DataModelling()
    datamodelling.read_data(input_path)
    datamodelling.clean()
    datamodelling.label_and_model()
    datamodelling.analysis()
