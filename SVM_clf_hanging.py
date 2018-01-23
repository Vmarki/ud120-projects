
# coding: utf-8

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 09:39:10 2018

@author: VB8625
"""

#!/usr/bin/python

import sys
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
#from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#I want to start with all numeric feature and select best or create new later
features_list = ['poi','bonus','deferral_payments', 'deferred_income', 
'director_fees', 'exercised_stock_options', 'expenses', 'loan_advances', 
'long_term_incentive', 'other', 'restricted_stock', 'restricted_stock_deferred', 
'salary', 'total_payments', 'total_stock_value', 'from_messages', 
'from_poi_to_this_person', 'from_this_person_to_poi', 'shared_receipt_with_poi', 
'to_messages'] 
# You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### Task 2: Remove outliers

# convert dict to dataframe
data_frame = pd.DataFrame.from_dict(data_dict, orient='index')
data_frame.replace('NaN', np.nan, inplace = True)

data_frame.info()

print "Number of persons of interest",len(data_frame[data_frame['poi']])


# Dataset has 146 entries with 21 features. 18 people where manually identified as persons of interest.

# <font color=gray>*Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]*</font>


import matplotlib.pyplot as plt
### Task 2: Remove outliers
#plot bonus vs salary to identify outliers
#get_ipython().magic('matplotlib inline')
data_frame.plot.scatter(x = 'salary', y = 'bonus')

# One value is very far from the rest by both salary and bonus. 


#find what key corresponds to the highest salary
data_frame['salary'].idxmax()


# Remove 'TOTAL' key from the data dictionary and data_frame.

#remove TOTAL outlier from data dictionary and data frame
data_dict.pop('TOTAL', 0)
data_frame.drop('TOTAL', inplace = True)
#plot bonus vs salary after removing 'TOTAL' outlier
data_frame.plot.scatter(x = 'salary', y = 'bonus')


#find who has maxixmum salary after 'TOTAL' is removed
data_frame['salary'].idxmax()


# Next highest salary data point is for Jeffrey Skilling, who is one of the main persons of interest, therefore I will all the rest of the data as apparent outliers may actually identify person of interest.

# <font color=gray>2. *What features did you end up using in your POI identifier, and what selection process did you use to pick them? In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “intelligently select features”, “properly scale features”]*</font>




### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

#replace 'NaN' with 0 in data dictionary
for key in my_dataset: 
    name = my_dataset[key] 
    for feature in features_list:
        if my_dataset[key][feature] == 'NaN':
            my_dataset[key][feature] = float(0)
            
#find the list of best features
from sklearn.feature_selection import SelectKBest, f_classif 
from sklearn.cross_validation import train_test_split 

def select_kbest_features(data_dict, features_list, k): 

    data = featureFormat(data_dict, features_list) 
    labels_train, features_train = targetFeatureSplit(data) 

    #print "features_list",features_list
    k_best = SelectKBest(f_classif, k=k) 
    k_best.fit(features_train, labels_train) 

    #print "k_best",k_best
    features_score_list = zip(features_list[1:], k_best.scores_) 
    #print "features_score_list",features_score_list
    sorted_feature_score_list = sorted(features_score_list, key=lambda x: x[1], reverse=True) 
    #print "sorted_feature_score_list",sorted_feature_score_list[:]
    return sorted_feature_score_list[:k]

kbest_features = select_kbest_features(my_dataset, features_list, 10)
print kbest_features

features = [val[0] for val in kbest_features][::-1]
print features
#scores = [val[1] for val in kbest_features][::-1]
#print scores

#plt.barh( range(len(kbest_features)), scores)
plt.barh(range(len(kbest_features)),  [val[1] for val in kbest_features][::-1])
plt.yticks(range(len(kbest_features)),  [val[0] for val in kbest_features][::-1])
plt.title('SelectKBest Feature Scores')
plt.show()


#add new field scaled_shared_receipt_with_poi
data_frame['scaled_shared_receipt_with_poi'] = ((data_frame['shared_receipt_with_poi']-
                                                min(data_frame['shared_receipt_with_poi']))/
                                                (max(data_frame['shared_receipt_with_poi'])
                                             -min(data_frame['shared_receipt_with_poi'])))
max_shared_rec = max(data_frame['shared_receipt_with_poi'])
min_shared_rec = min(data_frame['shared_receipt_with_poi'] ) 


#add scaled_shared_receipt_with_poi field to data dictionary
for key in my_dataset: 
    name = my_dataset[key] 
    if name['shared_receipt_with_poi'] != 'NaN' and name['shared_receipt_with_poi'] != 0:
        name['scaled_shared_receipt_with_poi'] = ((name['shared_receipt_with_poi']-min_shared_rec)/
                                                (max_shared_rec - min_shared_rec))
       # print name['scaled_shared_receipt_with_poi'] 
    else:
        name['scaled_shared_receipt_with_poi'] = float(0)


# <font color=gray>*As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.)*</font>

# I want ot combine all other numeric e-mail fields into one feature email_with_poi_ratio.

#add new field email_with_poi_ratio as relation of emails to and form poi to total number of emails available
data_frame['email_with_poi_ratio'] = ((data_frame['from_poi_to_this_person']+data_frame['from_this_person_to_poi'])/
                                      (data_frame['to_messages']+data_frame['from_messages']))
#print data_frame['email_with_poi_ratio']
#add email_with_poi_ratio to the data dictionary
#add scaled_shared_receipt_with_poi field to data dictionary
for key in my_dataset: 
    name = my_dataset[key] 
   # print name
    if (name['from_poi_to_this_person']== 'NaN' or name['from_this_person_to_poi']=='NaN'
        or name['to_messages']== 'NaN' or name['from_messages']=='NaN' or
       (name['to_messages'] == 0 and name['from_messages']==0)):
        name['email_with_poi_ratio'] = float(0)
    else:
        name['email_with_poi_ratio'] = (float(name['from_poi_to_this_person']+name['from_this_person_to_poi'])
                                        /float(name['to_messages']+name['from_messages']))
        #print name['email_with_poi_ratio']

from sklearn.metrics import accuracy_score
def test_data(clf, data, feature_list):
    labels, features = targetFeatureSplit( data )
    print "labels.size()",len(labels)
    #print "features.size()",len(features)
    ### training-testing split needed in regression, just like classification
    #from sklearn.cross_validation import train_test_split
    features_train, features_test, labels_train, labels_test = (train_test_split(features, 
                                                            labels, test_size=0.3, random_state=42))
    print "features_train.size()",len(features_train)
    #print "features_test.size()",len(features_test)
    #print "labels_train.size()",len(labels_train)
    #print "labels_test.size()",len(labels_test)
    print "before fit"
    ### fit the classifier on the training features and labels
    clf.fit(features_train,labels_train)
    print "after fit"
    ### use the trained classifier to predict labels for the test features
    pred = clf.predict(features_test)
    print "after predict"
    ### calculate and return the accuracy on the test data
    ### this is slightly different than the example, 
    ### where we just print the accuracy
    ### you might need to import an sklearn module
    print "before accuracy"
    accuracy = accuracy_score(labels_test, pred, normalize = True)
    print "after accuracy"
    print accuracy
    return accuracy


features_list1 = ['poi',
'exercised_stock_options',
'total_stock_value',
'bonus',
'salary',
'deferred_income', 
'long_term_incentive', 
'restricted_stock', 
'total_payments', 
'shared_receipt_with_poi', 
#'loan_advances', 
'from_poi_to_this_person',  
'from_this_person_to_poi', 
'to_messages', 
'from_messages'
                 ]
print len(my_dataset)
data1 = featureFormat( my_dataset, features_list1, remove_any_zeroes=False)
print "data1 size",len(data1)
features_list2 = ['poi',
'exercised_stock_options',
'total_stock_value',
'bonus',
'salary',
'deferred_income', 
'long_term_incentive', 
'restricted_stock', 
'total_payments',
#'loan_advances',
'email_with_poi_ratio', 
'scaled_shared_receipt_with_poi'] 

data2 = featureFormat( my_dataset, features_list2, remove_any_zeroes=False)
print "data size2",len(data2)


#create SVM clissifier
from sklearn.svm import SVC
#from sklearn.metrics import accuracy_score
clf = SVC(kernel="linear")

print ("test of accuracy original data with SVM classifier",
 test_data(clf, data1, features_list1))
print ("test of accuracy with new features data with SVM classifier", 
       test_data(clf, data2, features_list2))


#create Decision Tree clissifier
from sklearn import tree

clf = tree.DecisionTreeClassifier()

print ("test of accuracy original data with Decision Tree classifier", 
       test_data(clf, data1, features_list1))
print ("test of accuracy with new features data with Decision Tree classifier",
       test_data(clf, data2, features_list2))



### import the sklearn module for GaussianNB
from sklearn.naive_bayes import GaussianNB
### create classifier
clf = GaussianNB()
print ("test of accuracy original data with Naive Bayes classifier",
       test_data(clf, data1, features_list1))
print ("test of accuracy with new features data with Naive Bayes classifier",
       test_data(clf, data2, features_list2))

