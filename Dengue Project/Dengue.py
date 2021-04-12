"""
Classify patients in 3 levels of Dengue severity, according to the symptons assessed in hospital
0 for absence of symptons
1 for presence of symptons
SD: Severe Dengue
DwS: Dengue with warning signs
DnoWS: Dengue without warning signs

"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

DF = pd.read_csv('db52017.csv',delimiter=';')
#print(DF)
DFcol = DF.columns
num_lin = len(DF['idpac'])
print(num_lin)
#print(dir(DF))
DF = DF._drop_axis('idpac',axis=1)
DF = DF._drop_axis('days',axis=1)
DF = DF._drop_axis('age',axis=1)

c_pearson = DF.corr()
fig = plt.figure()
sns.heatmap(c_pearson,cmap='inferno',annot=True)
## it was seen in the heatmap of the correlation that the correlation of pvom.ws x nau.vom is 0.92. So I removed pvom.ws from dataset.
DF = DF._drop_axis('pvom.ws',axis=1)
print(f"Dataset without pvom.ws: {DF}")
print(" ")

from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
#k = 5 #number top features of Chi-2
chi2_sel = SelectKBest(score_func = chi2,k=5)
chi2_fit = chi2_sel.fit(DF.iloc[:,0:-1],DF['classf'])
chi2_scores = chi2_fit.scores_
X_new=chi2_sel.fit_transform(DF.iloc[:,0:-1],DF['classf'])

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X_new,DF['classf'],test_size=0.8,random_state=101)

#K-NN
from sklearn.neighbors import KNeighborsClassifier
error_rate = []
for i in range(4,30):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    pred = knn.predict(x_test)
    error_rate.append(np.mean(y_test!= pred))
plt.figure()
plt.plot(range(4,30),error_rate,color='blue',linestyle='dashed',marker='o',markersize=10)
plt.xlabel('K')
plt.ylabel('taxa de erro')
plt.title('taxa de erro vs valor de k')
plt.show()
#k = 9 is the one that give the lowest error rate
knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(x_train,y_train)
knn_pred = knn.predict(x_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test, knn_pred))
print(confusion_matrix(y_test, knn_pred))

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)
dt_pred = dt.predict(x_test)
print(classification_report(y_test, dt_pred))
print(confusion_matrix(y_test, dt_pred))

#Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train,y_train)
rf_pred = rf.predict(x_test)
print(classification_report(y_test, rf_pred))
print(confusion_matrix(y_test, rf_pred))

#Naïve Bayes
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(abs(x_train),y_train)
nb_pred = nb.predict(x_test)
print(classification_report(y_test, nb_pred))
print(confusion_matrix(y_test, nb_pred))

#SVM
from sklearn.svm import SVC
svm = SVC()
svm.fit(x_train,y_train)
svm_pred = svm.predict(x_test)
print(classification_report(y_test, svm_pred))
print(confusion_matrix(y_test, svm_pred))
