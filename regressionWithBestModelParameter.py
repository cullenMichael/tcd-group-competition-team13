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


def changeSizeOfCity(dataset):
    dataset['Small City'] = dataset['Size of City'] < 3000
    dataset['Small City'] = np.log(dataset['Size of City'])
    dataset.pop('Size of City')
    return dataset


def work_experience(dataset):
    # work_mean = dataset['Work Experience in Current Job [years]'].mean()
    dataset['Work Experience in Current Job [years]'] = dataset['Work Experience in Current Job [years]'].replace('#NUM!', 1)
    dataset['Work Experience in Current Job [years]'] = np.array(dataset['Work Experience in Current Job [years]'], dtype=np.float32)
    dataset['Work Experience in Current Job [years]'] = dataset['Work Experience in Current Job [years]'].replace(0, 1)
    # dataset['Work Experience in Current Job [years]'] = np.log(dataset['Work Experience in Current Job [years]'])
    return dataset


def processAdditionToSalary(dataset):
    dataset['Yearly Income in addition to Salary (e.g. Rental Income)'] = dataset['Yearly Income in addition to Salary (e.g. Rental Income)'].map(lambda x: x.rstrip(' EUR'))
    dataset['Yearly Income in addition to Salary (e.g. Rental Income)'] = dataset['Yearly Income in addition to Salary (e.g. Rental Income)'].astype('float')
    dataset['Yearly Income in addition to Salary (e.g. Rental Income)'] = dataset['Yearly Income in addition to Salary (e.g. Rental Income)'].replace(0, 1)
    # dataset['Yearly Income in addition to Salary (e.g. Rental Income)'] = np.log(dataset['Yearly Income in addition to Salary (e.g. Rental Income)'])
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
    dataset['Gender'] = dataset['Gender'].fillna(method='bfill')
    # dataset["Gender"] = dataset["Gender"].replace(np.NaN, "unknown")
    return dataset


def satisfaction(dataset):
    dataset['Satisfation with employer'] = dataset['Satisfation with employer'].fillna(method='bfill')
    dataset['Satisfation with employer'] = dataset['Satisfation with employer'].replace('Unhappy', 0)
    dataset['Satisfation with employer'] = dataset['Satisfation with employer'].replace('Average', 1)
    dataset['Satisfation with employer'] = dataset['Satisfation with employer'].replace('Somewhat Happy', 2)
    dataset['Satisfation with employer'] = dataset['Satisfation with employer'].replace('Happy', 3)
    dataset['Satisfation with employer'] = dataset['Satisfation with employer'].astype('int')
    return dataset


# def age(dataset):
#     dataset['Age'] = np.log(dataset['Age'])
#     return dataset


def year(dataset):
    dataset['Year of Record'] = dataset['Year of Record'].fillna(method='bfill')
    # dataset["Year of Record"] = dataset["Year of Record"].replace(np.nan, "MISSING")
    return dataset


def bodyHeight(dataset):
    return dataset


def profession(dataset):
    # dataset['Profession'].fillna('none', inplace=True)
    dataset['Profession'] = dataset['Profession'].fillna(method='bfill')
    return dataset


def housing(dataset):
    dataset["Housing Situation"] = dataset["Housing Situation"].replace(0, 0)
    dataset["Housing Situation"] = dataset["Housing Situation"].replace("0", 0)
    dataset["Housing Situation"] = dataset["Housing Situation"].replace("nA", np.nan)
    dataset['Housing Situation'] = dataset['Housing Situation'].fillna(method='bfill')



    # dataset["Housing Situation"] = dataset["Housing Situation"].replace('Castle', 0)
    # dataset["Housing Situation"] = dataset["Housing Situation"].replace('Large House', 0)
    # dataset["Housing Situation"] = dataset["Housing Situation"].replace('Medium House', 0)
    # dataset["Housing Situation"] = dataset["Housing Situation"].replace('Small House', 0)
    # dataset["Housing Situation"] = dataset["Housing Situation"].replace('Large Apartment', 3)
    # dataset["Housing Situation"] = dataset["Housing Situation"].replace('Medium Apartment', 2)
    # dataset["Housing Situation"] = dataset["Housing Situation"].replace('Small Apartment', 1)
    # dataset["Housing Situation"] = dataset["Housing Situation"].astype('int')

    # arr = np.zeros(len(dataset['Housing Situation']))
    # arr = arr[dataset["Housing Situation"] == 'Large Apartment']=1
    # arr = arr[dataset["Housing Situation"] == 'Medium Apartment']=2
    # arr = arr[dataset["Housing Situation"] == 'Small Apartment']=3

    # dataset.pop('Housing Situation')

    return dataset


def crime(dataset):
    dataset['C']


def hair(dataset):
    dataset['Hair Color'] = dataset['Hair Color'].replace(0, "Unknown")
    dataset['Hair Color'] = dataset['Hair Color'].replace('0', "Unknown")
    dataset['Hair Color'] = dataset['Hair Color'].replace(np.nan, "Unknown")
    return dataset


def main():

    dataset = pd.read_csv('tcd-ml-1920-group-income-train.csv')



    train = dataset.copy()

    train = train[:1044560]

    train = train.drop_duplicates(subset='Instance', keep='first',
                                  inplace=False)

    # train = train[train['Year of Record'] > 1980]

    y = np.log(train['Total Yearly Income [EUR]'])

    train = train.drop(columns=['Instance',
                                'Total Yearly Income [EUR]',
                                'Hair Color',
                                'Housing Situation',
                                'Wears Glasses',
                                ])

    train = bodyHeight(train)
    train = changeSizeOfCity(train)
    train = degree(train)
    train = genderCleaning(train)
    train = profession(train)
    train = satisfaction(train)
    train = work_experience(train)
    train = processAdditionToSalary(train)
    # train = housing(train)

    # import seaborn as sns
    # g = sns.pairplot(train[['Age', 'Housing Situation', 'Year of Record', 'Satisfation with employer', 'Yearly Income in addition to Salary (e.g. Rental Income)', 'Work Experience in Current Job [years]']])
    # plt.show()

    print ("Start Dummies")
    train = pd.get_dummies(train, columns=['Gender',
                                           'Country',
                                           'University Degree',
                                           'Profession'], drop_first=False)
    print "End Dummies"

    regressor = CatBoostRegressor(od_type='IncToDec')

    X_train, X_test, y_train, y_test = train_test_split(
        train, y, test_size=0.2)

    X_training, X_val, y_training, y_val = train_test_split(
        X_train, y_train, test_size=0.2)

    eval_dataset = Pool(X_val,
                        y_val)

    print ('Fitting')

    parameters = {'depth'         : sp_randInt(4, 11),
                  'learning_rate' : sp_randFloat(),
                  'iterations'    : sp_randInt(700, 800)
                 }

    randm = RandomizedSearchCV(estimator=regressor, param_distributions = parameters,
                               cv = 4, n_iter = 10, n_jobs=9)
    randm.fit(X_train, y_train)
    # Results from Random Search
    print("\n========================================================")
    print(" Results from Random Search ")
    print("========================================================")

    print("\n s:\n", randm.best_estimator_)

    print("\n The best score across ALL searched params:\n", randm.best_score_)

    print("\n The best parameters across ALL searched params:\n",randm.best_params_)

    # print("\n ========================================================")

    regressor = CatBoostRegressor(iterations=randm.best_params_['iterations'],
                                  learning_rate=randm.best_params_['learning_rate'],
                                  depth=randm.best_params_['depth'],
                                  od_type='IncToDec',
                                  use_best_model=True)


    # print ('Fitting')
    # regressor.fit(X_train, y_train)
    # print ('Predicting Locally')
    # y_pred = regressor.predict(X_test)
    # print('MAE is: {}'.format(mean_absolute_error(np.exp(y_test), np.exp(y_pred))))

    test_dataset = pd.read_csv(
        'tcd-ml-1920-group-income-test.csv')

    predict_X = test_dataset
    X_train = train
    y_train = y

    predict_X = predict_X.drop(columns=['Instance',
                                'Hair Color',
                                'Total Yearly Income [EUR]',
                                'Housing Situation',
                                'Wears Glasses',
                                ])

    predict_X = bodyHeight(predict_X)
    predict_X = changeSizeOfCity(predict_X)
    predict_X = degree(predict_X)
    predict_X = genderCleaning(predict_X)
    predict_X = profession(predict_X)
    predict_X = work_experience(predict_X)
    predict_X = processAdditionToSalary(predict_X)
    predict_X = year(predict_X)
    predict_X = satisfaction(predict_X)

    predict_X = pd.get_dummies(predict_X, columns=['Gender', 'Profession',
                                            'Country',
                                            'University Degree'], drop_first=False)

    X_train, predict_X = train.align(predict_X , join='outer', axis=1, fill_value=0)

    print 'Fitting Test Data'
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
