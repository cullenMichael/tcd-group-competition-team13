import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from category_encoders import *
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, MinMaxScaler,
                                   OneHotEncoder, OrdinalEncoder,
                                   StandardScaler)
from sklearn.utils import shuffle


def changeSizeOfCity(dataset):
    dataset['Small City'] = dataset['Size of City'] < 3000
    dataset['Small City'] = np.log(dataset['Size of City'])
    dataset.pop('Size of City')
    return dataset


def work_experience(dataset):
    dataset['Work Experience in Current Job [years]'] = dataset['Work Experience in Current Job [years]'].replace('#NUM!', 0, inplace= True)
    dataset['Work Experience in Current Job [years]'] = np.log(dataset['Work Experience in Current Job [years]'].astype('float'))

    return dataset


def processAdditionToSalary(dataset):
    dataset['Yearly Income in addition to Salary (e.g. Rental Income)'] = dataset['Yearly Income in addition to Salary (e.g. Rental Income)'].map(lambda x: x.rstrip(' EUR'))
    dataset['Yearly Income in addition to Salary (e.g. Rental Income)'] = np.log(dataset['Yearly Income in addition to Salary (e.g. Rental Income)'].astype('float'))

    return dataset


def degree(dataset):
    dataset["University Degree"] = dataset["University Degree"].replace(np.nan, "MISSING")
    dataset["University Degree"] = dataset["University Degree"].replace(0, "MISSING")
    dataset["University Degree"] = dataset["University Degree"].replace("0", "MISSING")
    return dataset


def genderCleaning(dataset):
    dataset["Gender"] = dataset["Gender"].replace("f", "female")
    dataset["Gender"] = dataset["Gender"].replace("0", "unknown")
    dataset["Gender"] = dataset["Gender"].replace(0, "unknown")
    dataset["Gender"] = dataset["Gender"].replace(np.NaN, "unknown")
    return dataset


def satisfaction(dataset):
    dataset['Satisfation with employer'] = dataset['Satisfation with employer'].replace(np.nan, "MISSING")
    return dataset


def age(dataset):
    dataset['Age'] = np.log(dataset['Age'])
    return dataset


def year(dataset):
    dataset["Year of Record"] = dataset["Year of Record"].replace(np.nan, "MISSING")
    return dataset


def bodyHeight(dataset):
    return dataset


def profession(dataset):
    dataset['Profession'].fillna('none', inplace=True)
    return dataset


def housing(dataset):
    dataset["Housing Situation"] = dataset["Housing Situation"].replace(0, "Unknown")
    dataset["Housing Situation"] = dataset["Housing Situation"].replace("0", "Unknown")
    dataset["Housing Situation"] = dataset["Housing Situation"].replace("nA", "Unknown")
    return dataset


def main():

    dataset = pd.read_csv('tcd-ml-1920-group-income-train.csv')

    train = dataset

    train = train[:1044560]
    train = train.drop_duplicates(subset='Instance', keep='first',
                                  inplace=False)

    y = np.log(train['Total Yearly Income [EUR]'])

    train = train.drop(columns=['Instance',
                                'Hair Color',
                                'Crime Level in the City of Employement',
                                'Total Yearly Income [EUR]',
                                ])
    train = bodyHeight(train)
    train = changeSizeOfCity(train)
    train = degree(train)
    train = genderCleaning(train)
    train = profession(train)
    train = work_experience(train)
    train = processAdditionToSalary(train)
    train = housing(train)

    # output = pd.read_csv('out.csv')
    train.to_csv("out.csv")

    train = pd.get_dummies(train, columns=['Gender', 'Profession',
                                           'Country',
                                           'University Degree',
                                           'Housing Situation',
                                           'Satisfation with employer'], drop_first=False)

    regressor = CatBoostRegressor()

    X_train, X_test, y_train, y_test = train_test_split(
        train, y, test_size=0.2)

    print ('Fitting')

    regressor.fit(X_train, y_train)
    print ('Predicting')
    y_pred = regressor.predict(X_test)
    print('MAE is: {}'.format(mean_absolute_error(np.exp(y_test), np.exp(y_pred))))

    # test_dataset = pd.read_csv(
    #     'tcd-ml-1920-group-income-test.csv')
    #
    # predict_X = test_dataset
    #
    #
    # predict_X = predict_X.drop(columns=['Instance',
    #                             'Hair Color',
    #                             'Crime Level in the City of Employement',
    #                             'Total Yearly Income [EUR]'
    #                             ])
    #
    # predict_X = bodyHeight(predict_X)
    # predict_X = changeSizeOfCity(predict_X)
    # predict_X = degree(predict_X)
    # predict_X = genderCleaning(predict_X)
    # predict_X = profession(predict_X)
    # predict_X = work_experience(predict_X)
    # predict_X = processAdditionToSalary(predict_X)
    #
    # predict_X = pd.get_dummies(predict_X, columns=['Gender', 'Profession',
    #                                         'Country',
    #                                         'University Degree',
    #                                         'Housing Situation',
    #                                         'Satisfation with employer'], drop_first=False)
    # train, predict_X = train.align(predict_X , join='outer', axis=1, fill_value=0)
    #
    # regressor.fit(X_train, y_train)
    #
    # pred2 = regressor.predict(predict_X)
    # # print(pred2)
    # # Write to file
    # # pred2 = pred2 + additionSal
    # output = pd.read_csv('tcd-ml-1920-group-income-submission.csv')
    # instance = output['Instance']
    # output.pop('Instance')
    # a = pd.DataFrame.from_dict({
    #     'Instance': instance,
    #     'Total Yearly Income [EUR]': np.exp(pred2)
    # })
    # a.to_csv("tcd-ml-1920-group-income-submission.csv", index=False)



if __name__ == "__main__":
    main()
