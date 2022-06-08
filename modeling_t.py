import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier 


filepath="/Data-all(data expansion with 33).xlsx" 
Data_ = pd.read_excel(filepath)
#print (Data_[:2])

ndarray = Data_.values
#print (type(ndarray))
print(ndarray.shape)

#///////////////////////////////////////////////////

X_o = ndarray[::,1:5]
y_o = ndarray[::,-1]

per = np.random.permutation(X_o.shape[0])
X = X_o[per, :]
y = y_o[per]

# minmax_scale = preprocessing.MinMaxScaler(feature_range=(0,1))
# scaledFeatures = minmax_scale.fit_transform(X)

z_score_scale = preprocessing.StandardScaler()
scaledFeatures = z_score_scale.fit_transform(X)


# K-fold
k = 10
num_val_samples = len(scaledFeatures) // k
scores_knn = []
scores_logre = []
scores_svm = []
scores_RFC = []

for i in range(k):
    print('processing fold #', i)
    
    # Prepare the validation data: data from partition # k
    val_data = scaledFeatures[i * num_val_samples: (i + 1) * num_val_samples]   
    val_targets = y[i * num_val_samples: (i + 1) * num_val_samples]
    
    # Prepare the training data: data from all other partitions
    partial_train_data = np.concatenate(                                    
        [scaledFeatures[:i * num_val_samples],
         scaledFeatures[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [y[:i * num_val_samples],
         y[(i + 1) * num_val_samples:]],
        axis=0)


    print("[INFO] start training model knn …")
    knn = KNeighborsClassifier(n_neighbors = 5)
    knn.fit(partial_train_data, partial_train_targets)
    y_pte1_knn = knn.predict(val_data)
    # # printing the results 
    print("[INFO] Reporting …")
    # print(classification_report(val_targets, y_pte1_knn))
    print ('Confusion Matrix :') 
    print(confusion_matrix(val_targets, y_pte1_knn))
    score_knn = accuracy_score(val_targets, y_pte1_knn)
    scores_knn.append(score_knn)



    print("[INFO] start training model Logistic Regression …")
    logre = LogisticRegression()
    logre.fit(partial_train_data, partial_train_targets)
    y_pte1_logre = logre.predict(val_data)
    # printing the results 
    print("[INFO] Reporting …")
    # print(classification_report(val_targets, y_pte1_logre))
    print ('Confusion Matrix :') 
    print(confusion_matrix(val_targets, y_pte1_logre))
    score_logre = accuracy_score(val_targets, y_pte1_logre)
    scores_logre.append(score_logre)


    # train a Linear SVM on the data
    print("[INFO] start training model SVM …")
    # svm = svm.LinearSVC()
    # svm = LinearSVC(C=100.0, random_state=42)
    svm_ = svm.SVC(kernel='rbf',C=1, gamma='auto')
    svm_.fit(partial_train_data, partial_train_targets)
    y_pte1_svm = svm_.predict(val_data)
    # # printing the results 
    print("[INFO] Reporting …")
    # print(classification_report(val_targets, y_pte1_svm))
    print ('Confusion Matrix :') 
    print(confusion_matrix(val_targets, y_pte1_svm))
    score_svm = accuracy_score(val_targets, y_pte1_svm)
    scores_svm.append(score_svm)
    
    
    print("[INFO] start training model Random Forest Classification …")
    RFC=RandomForestClassifier() 
    RFC.fit(partial_train_data, partial_train_targets)
    y_pte1_RFC = RFC.predict(val_data)
    #printing the results 
    print("[INFO] Reporting …")
    # print(classification_report(val_targets, y_pte1_RFC))
    print ('Confusion Matrix :') 
    print(confusion_matrix(val_targets, y_pte1_RFC))
    score_RFC = accuracy_score(val_targets, y_pte1_RFC)
    scores_RFC.append(score_RFC)
    
print('Results:')
print('knn accurracy: ', np.mean(scores_knn))
print('Logistic Regression accurracy: ', np.mean(scores_logre))
print('svm accurracy: ', np.mean(scores_svm))
print('Random Forest accurracy: ', np.mean(scores_RFC))
