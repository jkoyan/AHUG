__author__ = 'JKoyan'

# This code was forked from udacity's code repository @ https://github.com/udacity/ud120-projects

from sklearn.naive_bayes import GaussianNB
import numpy as np
from StringIO import StringIO
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split

from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture



features_train,labels_train,features_test,labels_test = makeTerrainData(1000)

clf = GaussianNB()
clf.fit(features_train,labels_train)
labels_pred = clf.predict(features_test)

# evaluate the performance of the classifier
score = accuracy_score(labels_test,labels_pred,normalize=False)
accuracy = score/float(len(labels_test))

print('The classifier was able to label {0} no of test observations correctly out of total {1} observations, with an accuracy of {2} %.'.format(score,len(labels_test),accuracy*100.0) )
prettyPicture(clf,features_test,labels_test)

