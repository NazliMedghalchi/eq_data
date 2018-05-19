import pandas as pd
import numpy as np
import sklearn as skl
from sklearn import preprocessing, neighbors, cross_validation

import os, sys
import datetime
from pandas import DataFrame


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

    def label_and_model(self):
        class_path = os.path.join(os.getcwd(), 'data/POIList.csv')
        lables_data = np.array(pd.read_csv(class_path))
        features_data = np.array(self.data[['Latitude', 'Longitude']])

        # training and testing data
        features_train, features_test, lables_data_train, lables_data_test = \
            cross_validation.train_test_split(features_data, lables_data, test_size=0.3)

        # classifier is KNN
        neightbours = neighbors.KNeighborsClassifier(n_neighbors=4)

        # Model based on KNN
        neightbours.fit(features_train, lables_data_train)

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
