from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def grabVecs(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)


train_vecs = grabVecs('train_vecs.txt')
y_train = grabVecs('y_train.txt')
test_vecs = grabVecs('test_vecs.txt')
y_test = grabVecs('y_test.txt')

lr = SGDClassifier(loss='log', penalty='l1')
lr.fit(train_vecs, y_train)
print 'Test Accuracy of Logistic: %.2f' % lr.score(test_vecs, y_test)
print 'Train Accuracy of Logistic: %.2f' % lr.score(train_vecs, y_train)
'''
pred_probas = lr.predict_proba(test_vecs)[:, 1]

plt.figure(1)
fpr, tpr, _ = roc_curve(y_test, pred_probas)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='area = %.2f' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc='lower right')
plt.show()
'''
svm = SVC(C=50, gamma=1, probability=True)
svm.fit(train_vecs, y_train)
print 'Test Accuracy of SVM: %.2f' % svm.score(test_vecs, y_test)
print 'Train Accuracy of SVM: %.2f' % svm.score(train_vecs, y_train)
print metrics.classification_report(y_train, svm.predict(train_vecs))

pred_probas = svm.predict_proba(test_vecs)[:, 1]

plt.figure(2)
fpr, tpr, _ = roc_curve(y_test, pred_probas)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='area = %.2f' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc='lower right')
plt.show()