import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn import metrics, model_selection, preprocessing, tree
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (LabelEncoder, MinMaxScaler, OneHotEncoder,
                                   StandardScaler)


def imputer(dataset):
    dataset = dataset.drop(columns=['Instance', 'Hair Color'])
    dataset['Gender'] = dataset['Gender'].replace(['0'], np.NaN)
    dataset['Gender'] = dataset['Gender'].replace(['unknown'], np.NaN)
    dataset['University Degree'] = dataset['University Degree'].replace(['0'], np.NaN)
    dataset['Country'] = dataset['Country'].replace(['0'], np.NaN)

    features_numeric = ['Year of Record', 'Age', 'Size of City', 'Body Height [cm]', 'Wears Glasses']
    features_categoric = ['University Degree', 'Gender', 'Country', 'Profession']

    imputer_numeric = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
    ])

    imputer_categoric = Pipeline(
        steps=[('imputer',
                SimpleImputer(strategy='most_frequent'))])

    preprocessor = ColumnTransformer(transformers=[('imputer_numeric',
                                                    imputer_numeric,
                                                    features_numeric),
                                                   ('imputer_categoric',
                                                    imputer_categoric,
                                                    features_categoric)])

    preprocessor.fit(dataset)
    return pd.DataFrame(preprocessor.transform(dataset),
                        columns=['Year of Record', 'Age', 'Size of City', 'Body Height [cm]', 'Wears Glasses', 'University Degree', 'Gender', 'Country', 'Profession'])














def main():
    df = pd.read_csv('tcd-ml-2019-20-income-prediction-training-with-labels.csv')








if __name__== "__main__":
    main()
