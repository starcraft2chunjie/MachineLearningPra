import numpy as np 
import pandas as pd 
trainData = pd.read_csv('train.csv')
testData = pd.read_csv('test.csv')

#drop some data class
trainData = trainData.drop(['Cabin'], axis = 1)
testData = testData.drop(['Cabin'], axis = 1)
trainData = trainData.drop(['Ticket'], axis = 1)
testData = testData.drop(['Ticket'], axis = 1)

#fill in the age value
combine = [trainData, testData]
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand = False)
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Donna'], 'Rare')
    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
    dataset['Title'] = dataset['Title'].replace('Mile', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Royal': 5, 'Rare': 6}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
age_title_mapping = {1: 30, 2: 20, 3: 40, 4: 3, 5: 40, 6: 40}
trainData['Title'] = trainData['Title'].map(age_title_mapping)
testData['Title'] = testData['Title'].map(age_title_mapping)
#fill in the missing value
#fill in the Embarked value
trainData = trainData.fillna({'Embarked':'S'})
testData = testData.fillna({'Embarked': 'S'})
#sort the ages into logical categories
trainData['Age'] = trainData['Age'].fillna(0)
testData['Age'] = testData['Age'].fillna(0)
for x in range(len(trainData['Age'])):
    if trainData['Age'][x] == 0:
        trainData['Age'][x] = trainData['Title'][x]
bins = [0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Baby', 'Child', 'Teenager', 'Student', 'Young adult', 'Adult', 'Senior']
trainData['AgeGroup'] = pd.cut(trainData['Age'], bins, labels = labels)
testData['AgeGroup'] = pd.cut(testData['Age'], bins, labels = labels)

age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young adult': 5, 'Adult': 6, 'Senior': 7}

#convert the age group into a numerical value
trainData['AgeGroup'] = trainData['AgeGroup'].map(age_mapping)
testData['AgeGroup'] = testData['AgeGroup'].map(age_mapping)

#drop the Age value for now
trainData = trainData.drop(['Age'], axis = 1)
testData = testData.drop(['Age'], axis = 1)

#drop the Name value since it contains no useful information
trainData = trainData.drop(['Name'], axis = 1)
testData = testData.drop(['Name'], axis = 1)

#map each Sex value into a numerical value
sex_mapping = {'male' : 0, 'female' : 1}
trainData['Sex'] = trainData['Sex'].map(sex_mapping)
testData['Sex'] = testData['Sex'].map(sex_mapping)

#map each Embarked value into a numerical value
embarked_mapping = {'S': 1, 'C': 2, 'Q': 3}
trainData['Embarked'] = trainData['Embarked'].map(embarked_mapping)
testData['Embarked'] = testData['Embarked'].map(embarked_mapping)

for x in range(len(testData['Fare'])):
    if pd.isnull(testData['Fare'][x]):
        pclass = testData['Pclass'][x]
        testData['Fare'][x] = round(trainData[trainData['Pclass'] == pclass]['Fare'].mean(), 4)

trainData['FareBand'] = pd.qcut(trainData['Fare'], 4, labels = [1, 2, 3, 4])
testData['FareBand'] = pd.qcut(trainData['Fare'], 4, labels = [1, 2, 3, 4])
#drop Fare values
trainData = trainData.drop(['Fare'], axis = 1)
testData = testData.drop(['Fare'], axis = 1)

predictors = trainData.drop(['Survived', 'PassengerId'], axis = 1)
target = trainData['Survived']