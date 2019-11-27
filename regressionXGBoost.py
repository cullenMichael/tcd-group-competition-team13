import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from category_encoders import *
from scipy.stats import randint as sp_randInt
from scipy.stats import uniform as sp_randFloat
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, MinMaxScaler,
                                   OneHotEncoder, OrdinalEncoder,
                                   StandardScaler)
from sklearn.utils import shuffle
import math
from xgboost import XGBRegressor

'''
    Adds new column for Cities less than 3000
'''
def changeSizeOfCity(dataset):
    dataset['Small City'] = dataset['Size of City'] < 3000
    dataset['Size of City'] = np.log(dataset['Size of City'])
    return dataset


'''
    Gets rid of '#NUM!', converts to float array and scales around the mean
'''
def work_experience(dataset):
    dataset['Work Experience in Current Job [years]'] = dataset['Work Experience in Current Job [years]'].replace('#NUM!', 1)
    dataset['Work Experience in Current Job [years]'] = np.array(dataset['Work Experience in Current Job [years]'], dtype=np.float32)
    #mean = dataset['Work Experience in Current Job [years]'].mean()
    #dataset['Work Experience in Current Job [years]'] = dataset['Work Experience in Current Job [years]'] - mean
    dataset['Work Experience in Current Job [years]'] = dataset['Work Experience in Current Job [years]'].replace(0, 1)
    dataset['Work Experience in Current Job'] = np.log(dataset['Work Experience in Current Job [years]'])
    dataset.pop('Work Experience in Current Job [years]')
    return dataset


'''
    Strips 'EUR', converts to float and scales with log
'''
def processAdditionToSalary(dataset):
    dataset['Yearly Income in addition to Salary (e.g. Rental Income)'] = dataset['Yearly Income in addition to Salary (e.g. Rental Income)'].map(lambda x: x.rstrip(' EUR'))
    dataset['Yearly Income in addition to Salary (e.g. Rental Income)'] = dataset['Yearly Income in addition to Salary (e.g. Rental Income)'].astype('float')
    dataset['Yearly Income in addition to Salary (e.g. Rental Income)'] = dataset['Yearly Income in addition to Salary (e.g. Rental Income)'].replace(0, 1)
    dataset['Yearly Income in addition to Salary (e.g. Rental Income)'] = np.log(dataset['Yearly Income in addition to Salary (e.g. Rental Income)'])
    return dataset


'''
    Replaces nan and 0 values with MISSING
'''
def degree(dataset):
    dataset["University Degree"] = dataset["University Degree"].replace(np.nan, "MISSING")
    dataset["University Degree"] = dataset["University Degree"].replace(0, "MISSING")
    dataset["University Degree"] = dataset["University Degree"].replace("0", "MISSING")
    return dataset


'''
    Replaces f with Female, Replaces 0 and nan with unknown, fills unknown values based on their Body Height
'''
def genderCleaning(dataset):
    dataset["Gender"] = dataset["Gender"].replace("f", "female")
    dataset["Gender"] = dataset["Gender"].replace("0", "unknown")
    dataset["Gender"] = dataset["Gender"].replace(0, "unknown")
    dataset["Gender"] = dataset["Gender"].replace(np.NaN, "unknown")
    #dataset['Gender'] = dataset['Gender'].fillna(method='bfill')

    gender_height = dataset.groupby('Gender').median()[['Body Height [cm]']]
    male= gender_height.loc["male"]["Body Height [cm]"]
    female= gender_height.loc["female"]["Body Height [cm]"]
    for (i, row) in dataset.iterrows():
        if row["Gender"] == "unknown":
            if abs(male-row["Body Height [cm]"]) < abs(female - row["Body Height [cm]"]):
                dataset.set_value(i,"Gender", "male")
            else:
                dataset.set_value(i,"Gender", "female")
    print("Finished Gender")

    return dataset


'''
    Backwards Fill the missing values. Ordinal encode the data and convert to int
'''
def satisfaction(dataset):
    dataset['Satisfation with employer'] = dataset['Satisfation with employer'].fillna(method='bfill')
    dataset['Satisfation with employer'] = dataset['Satisfation with employer'].replace('Unhappy', 0)
    dataset['Satisfation with employer'] = dataset['Satisfation with employer'].replace('Average', 1)
    dataset['Satisfation with employer'] = dataset['Satisfation with employer'].replace('Somewhat Happy', 2)
    dataset['Satisfation with employer'] = dataset['Satisfation with employer'].replace('Happy', 3)
    dataset['Satisfation with employer'] = dataset['Satisfation with employer'].astype('int')
    return dataset

'''
    In test data, fill the years based on the Housing Situation (High correlation with Year)
'''
def year(dataset):
    #dataset['Year of Record'] = dataset['Year of Record'].fillna(method='bfill')
    dataset["Housing Situation"] = dataset["Housing Situation"].replace("nA", "Unknown")
    housing = dataset.groupby('Housing Situation').mean()[['Year of Record']]

    for (i, row) in dataset.iterrows():
        if math.isnan(row["Year of Record"]):
            dataset.set_value(i,"Year of Record", int(housing.loc[row["Housing Situation"]]["Year of Record"]))
    return dataset


'''
    Scale Body Height around the mean
'''
def bodyHeight(dataset):
    mean = dataset['Body Height [cm]'].mean()
    dataset['Body Height'] = dataset['Body Height [cm]'] - mean
    dataset.pop("Body Height [cm]")
    return dataset


'''
    Backwards Fill Profession and only take the first 5 characters for each profession.
'''
def profession(dataset):
    dataset['Profession'] = dataset['Profession'].fillna(method='bfill')
    dataset["Profession"] = dataset["Profession"].astype(str).str[:5]
    return dataset


'''
    Scales Crime Level around the mean
'''
def crime(dataset):
    mean = dataset['Crime Level in the City of Employement'].mean()
    dataset['Crime Level in the City of Employement'] = dataset['Crime Level in the City of Employement'] - mean
    return dataset


def main():

    dataset = pd.read_csv('tcd-ml-1920-group-income-train.csv')
    train = dataset.copy()
    train = train[:1044560]
    train = train.drop_duplicates(subset='Instance', keep='first',inplace=False)
    train = train.drop(columns = ['Instance'])
    train = train.drop_duplicates(inplace = False)
    y = np.log(train['Total Yearly Income [EUR]'])

    train = train.drop(columns=['Total Yearly Income [EUR]',
                                'Hair Color',
                                'Housing Situation',
                                'Wears Glasses',
                                ])

    train = changeSizeOfCity(train)
    train = degree(train)
    train = genderCleaning(train)
    train = bodyHeight(train)
    train = profession(train)
    train = satisfaction(train)
    train = work_experience(train)
    train = processAdditionToSalary(train)
    train = crime(train)

    #Encode using get dummies
    print ("Start Dummies")
    #train = pd.get_dummies(train, columns=['Gender',
    #                                       'Country',
    #                                       'University Degree',
    #                                       'Profession'], drop_first=True)
    te = TargetEncoder()
    train[['Gender','Country', 'Profession', 'University Degree']] = te.fit_transform(train[['Gender','Country', 'Profession', 'University Degree']], y)
    print ("End Dummies")

    regressor = XGBRegressor()

    X_train, X_test, y_train, y_test = train_test_split(
        train, y, test_size=0.2)

    X_training, X_val, y_training, y_val = train_test_split(
        X_train, y_train, test_size=0.2)

    eval_dataset = Pool(X_val,
                        y_val)

    print ('Fitting')
    parameters = {'depth'         : sp_randInt(3,4),
                  'learning_rate' : sp_randFloat(),
                  'iterations'    : sp_randInt(200, 300)
                 }

    randm = RandomizedSearchCV(estimator=regressor, param_distributions = parameters,
                               cv = 4, n_iter = 10, n_jobs=5)
    randm.fit(X_train, y_train)
    # Results from Random Search
    print("\n========================================================")
    print(" Results from Random Search ")
    print("========================================================")

    print("\n s:\n", randm.best_estimator_)

    print("\n The best score across ALL searched params:\n", randm.best_score_)

    print("\n The best parameters across ALL searched params:\n",randm.best_params_)

    regressor = CatBoostRegressor(iterations=randm.best_params_['iterations'],
                                  learning_rate=randm.best_params_['learning_rate'],
                                  depth=randm.best_params_['depth'],
                                  od_type='IncToDec',
                                  use_best_model=True)


    test_dataset = pd.read_csv(
        'tcd-ml-1920-group-income-test.csv')

    predict_X = test_dataset
    X_train = train
    y_train = y
    predict_y = predict_X.pop('Total Yearly Income [EUR]')

    predict_X = year(predict_X)
    predict_X = predict_X.drop(columns=['Instance',
                                'Hair Color',
                                #'Total Yearly Income [EUR]',
                                'Housing Situation',
                                'Wears Glasses',
                                ])


    predict_X = changeSizeOfCity(predict_X)
    predict_X = degree(predict_X)
    predict_X = genderCleaning(predict_X)
    predict_X = bodyHeight(predict_X)
    predict_X = profession(predict_X)
    predict_X = work_experience(predict_X)
    predict_X = processAdditionToSalary(predict_X)

    predict_X = satisfaction(predict_X)
    predict_X = crime(predict_X)

    #predict_X = pd.get_dummies(predict_X, columns=['Gender', 'Profession',
    #                                        'Country',
    #                                        'University Degree'], drop_first=True)
    predict_X[['Gender','Country', 'Profession', 'University Degree']] = te.transform(predict_X[['Gender','Country', 'Profession', 'University Degree']], predict_y)
    X_train, predict_X = train.align(predict_X , join='outer', axis=1, fill_value=0)

    print ('Fitting Test Data')
    regressor.fit(X_training, y_training,eval_set=eval_dataset)

    pred2 = regressor.predict(predict_X)
    output = pd.read_csv('tcd-ml-1920-group-income-submission.csv')
    instance = output['Instance']
    output.pop('Instance')
    a = pd.DataFrame.from_dict({
        'Instance': instance,
        'Total Yearly Income [EUR]': np.exp(pred2)
    })
    a.to_csv("tcd-ml-1920-group-income-submission.csv", index=False)


    y_pred = regressor.predict(X_test)
    print('MAE is: {}'.format(mean_absolute_error(np.exp(y_test), np.exp(y_pred))))


if __name__ == "__main__":
    main()
