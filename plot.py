import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
dataset = pd.read_csv(
    'tcd-ml-1920-group-income-train.csv')

train = dataset
#y = train.pop('Total Yearly Income [EUR]').values


#plt.scatter(train['Size of City'], train['Total Yearly Income [EUR]'])
#plt.title('Size of City')
#plt.show()
#Discard sizer of city below 1 x t=10^7

si =SimpleImputer(strategy='constant',fill_value='MISSING')
train[['Hair Color']] = si.fit_transform(train[['Hair Color']])
plt.scatter(train['Hair Color'], train['Total Yearly Income [EUR]'])
plt.title('Hair Color')
plt.show()
#possobly drop this

#Crime Level and Work Experience Similar
plt.scatter(train['Crime Level in the City of Employement'], train['Total Yearly Income [EUR]'])
plt.title('Crime Level')
plt.show()

train['Work Experience in Current Job [years]'].replace('#NUM!', np.nan, inplace= True)
plt.scatter(train['Work Experience in Current Job [years]'], train['Total Yearly Income [EUR]'])
plt.title('Work Experience')
plt.show()


#plt.scatter(train['Body Height [cm]'], train['Total Yearly Income [EUR]'])
#plt.title('Body Height')
#plt.show()
#Gaussian Distribution


train[['Gender']] = si.fit_transform(train[['Gender']])
train['Gender'].replace('f', 'female', inplace= True)
train['Gender'].replace('0', 'unknown', inplace= True)
plt.scatter(train['Gender'], train['Total Yearly Income [EUR]'])
plt.title('Gender')
plt.show()
#Possibly get rid of unknown/missing??

train[['Housing Situation']] = si.fit_transform(train[['Housing Situation']])
train['Housing Situation'].replace('nA', 'MISSING', inplace= True)
train['Housing Situation'].replace(0, 'MISSING', inplace= True)
plt.scatter(train['Housing Situation'], train['Total Yearly Income [EUR]'])
plt.title('Housing Situation')
plt.show()
#Seems to have a curve?? Possibly ordinal encoding??

train[['University Degree']] = si.fit_transform(train[['University Degree']])
plt.scatter(train['University Degree'], train['Total Yearly Income [EUR]'])
plt.title('University Degree')
plt.show()
#All people with any form of degree??

train[['Satisfation with employer']] = si.fit_transform(train[['Satisfation with employer']])
plt.scatter(train['Satisfation with employer'], train['Total Yearly Income [EUR]'])
plt.title('Satisfaction with Employer')
plt.show()

#This plot doesn't work - not sure why
'''
train[['Yearly Income in addition to Salary (e.g. Rental Income)']] = si.fit_transform(train[['Yearly Income in addition to Salary (e.g. Rental Income)']])
train['Yearly Income in addition to Salary (e.g. Rental Income)'] = train['Yearly Income in addition to Salary (e.g. Rental Income)'].map(lambda x: x.rstrip(' EUR'))
train['Yearly Income in addition to Salary (e.g. Rental Income)'].astype('float')
plt.scatter(train['Yearly Income in addition to Salary (e.g. Rental Income)'], train['Total Yearly Income [EUR]'])
plt.title('Yearly Income in addition to Salary (e.g. Rental Income)')
plt.show()
'''
plt.scatter(train['Year of Record'], train['Total Yearly Income [EUR]'])
plt.title('Year of Record')
plt.show()
#Exponential
