import numpy as np 
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=2017)

def recall(test, cm):
    row = cm[test, :]
    return cm[test, test] / row.sum()

def precision(test, cm):
    col = cm[:, test]
    return cm[test, test] / col.sum()

def statistics(digits, cm):
    for x in range(digits):
        print('---------------------')
        print('Digit:',x)
        print('Recall:', recall(x, cm))
        print('Precision', precision(x,cm))
        print('Fmeasure', (2*recall(x,cm)*precision(x,cm)) / (recall(x,cm) + precision(x,cm)))
        print('---------------------')
    

kVals = range(1, 30, 2)
accuracies = []
for k in range(1, 30, 2):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    score = knn.score(X_test, y_test)
    accuracies.append(score)
## 9 is the best
i = int(np.argmax(accuracies))
print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i],accuracies[i] * 100))

knn = KNeighborsClassifier(n_neighbors=i)
print('KNN score: %f' % knn.fit(X_train, y_train).score(X_test, y_test))
k_pred = knn.predict(X_test)
cmat = confusion_matrix(y_test, k_pred)
print(cmat)
statistics(10, cmat)

svm_classifier = svm.SVC(gamma=0.001)
print('SVM score: %f' % svm_classifier.fit(X_train, y_train).score(X_test, y_test))
s_pred = svm_classifier.predict(X_test)
svm_cmat = confusion_matrix(y_test, s_pred)
print(svm_cmat)
statistics(10, svm_cmat)

random_forest_classifier = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
print('Random forest score: %f' % random_forest_classifier.fit(X_train, y_train).score(X_test, y_test))
r_pred = random_forest_classifier.predict(X_test)
r_cmat = confusion_matrix(y_test, r_pred)
print(r_cmat)
statistics(10, r_cmat)