import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import seaborn as sns
from sklearn import linear_model, preprocessing
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

data = pd.read_csv('car.data')

#converting into integer values
le = preprocessing.LabelEncoder()
#making array for each of the columns
#doing with pd 'cause it's a lot easier

buying = le.fit_transform(list(data['buying']))
maint = le.fit_transform(list(data['maint']))
door = le.fit_transform(list(data['door']))
persons = le.fit_transform(list(data['persons']))
lug_boot = le.fit_transform(list(data['lug_boot']))
safety = le.fit_transform(list(data['safety']))
cls = le.fit_transform(list(data['class']))

predict = 'class'
#returns to us a numpy array, le.fit_transform

x = list(zip(buying,maint,door,persons,lug_boot,safety))
#converts all of it in one big list
y = list(cls)

#split, train, test

x_train,x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size = 0.1)

model = KNeighborsClassifier(n_neighbors=7)
#5 is numbe of the neighbours
model.fit(x_train, y_train)
acc = model.score(x_test,y_test)
print(acc)

predicted = model.predict(x_test)
names = ['unacc','acc','good','vrygood']
#names that classifier classifies things as
#it classifies 0-3, so we get the actual value

for x in range(len(predicted)):
    print('Predicted: ', names[predicted[x]], 'Data: ', x_test[x], 'Actual: ', names[y_test[x]])
    #n = model.kneighbors([x_test[x]], 9, True)
    #print("N: ", n) #better not do with matplotlib


'''
p = 'persons'
style.use('ggplot')
pyplot.scatter(data[p],data['buying'])
pyplot.xlabel(p)
pyplot.ylabel('Final Grade')
pyplot.show()'''