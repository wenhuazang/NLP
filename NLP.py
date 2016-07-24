import gensim
from gensim.models.doc2vec import Doc2Vec
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn import metrics, neighbors

LabeledSentence = gensim.models.doc2vec.LabeledSentence

import numpy as np
import os
import random
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

data_path = "review_polarity/txt_sentoken/"
pos_files = os.listdir(data_path + "pos/")
neg_files = os.listdir(data_path + "neg/")
pos_reviews = []
neg_reviews = []
unsup_reviews1 = []
unsup_reviews2 = []
unsup_reviews = []

for pos_file in pos_files[:900]:
    with open(data_path + "pos/" + pos_file, 'r') as infile:
        each = ''
        each_article = infile.readlines()
        for each_sentence in each_article:
            each += each_sentence
    pos_reviews.extend([each])
for neg_file in neg_files[:900]:
    with open(data_path + "neg/" + neg_file, 'r') as infile:
        each = ''
        each_article = infile.readlines()
        for each_sentence in each_article:
            each += each_sentence
    neg_reviews.extend([each])
for pos_file in pos_files[900:]:
    with open(data_path + "pos/" + pos_file, 'r') as infile:
        each = ''
        each_article = infile.readlines()
        for each_sentence in each_article:
            each += each_sentence
    unsup_reviews1.extend([each])
for neg_file in neg_files[900:]:
    with open(data_path + "neg/" + neg_file, 'r') as infile:
        each = ''
        each_article = infile.readlines()
        for each_sentence in each_article:
            each += each_sentence
    unsup_reviews2.extend([each])


# use 1 for positive sentiment, 0 for negative
y = np.concatenate((np.ones(len(pos_reviews)), np.zeros(len(neg_reviews))))

x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos_reviews, neg_reviews)), y, test_size=0.2)
unsup_reviews.extend(unsup_reviews1)
unsup_reviews.extend(unsup_reviews2)


# Do some very minor text preprocessing
def cleanText(corpus):
    punctuation = """.,?!:;(){}[]-_"""
    corpus = [z.lower().replace('\n', '') for z in corpus]
    corpus = [z.replace('<br />', ' ') for z in corpus]

    # treat punctuation as individual words
    for c in punctuation:
        corpus = [z.replace(c, ' ') for z in corpus]
    corpus = [z.split() for z in corpus]
    return corpus


x_train = cleanText(x_train)
x_test = cleanText(x_test)
unsup_reviews = cleanText(unsup_reviews)


# Gensim's Doc2Vec implementation requires each document/paragraph to have a label associated with it.
# We do this by using the LabeledSentence method. The format will be "TRAIN_i" or "TEST_i" where "i" is
# a dummy index of the review.
def labelizeReviews(reviews, label_type):
    labelized = []
    for i, v in enumerate(reviews):
        label = '%s_%s' % (label_type, i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized


x_train = labelizeReviews(x_train, 'TRAIN')
x_test = labelizeReviews(x_test, 'TEST')
unsup_reviews = labelizeReviews(unsup_reviews, 'UNSUP')

size = 400
iteration = 5

model = Doc2Vec.load_word2vec_format('vectors.bin', binary=True)


# Get training set vectors from our models
def getVecs(model, corpus, size):
    vecs = []
    for z in corpus:
        vec = np.zeros(size).reshape((1, size))
        count = 0.
        for each in np.array(model[z.words]):
            try:
                vec += each.reshape((1, size))
                count += 1.
            except KeyError:
                continue
        if count != 0:
            vec /= count
        vecs.extend(vec)
    return np.array(vecs)

train_vecs = getVecs(model, x_train, size)

# Construct vectors for test reviews
test_vecs = getVecs(model, x_test, size)


lr = SGDClassifier(loss='log', penalty='l1')
lr.fit(train_vecs, y_train)
print 'Test Accuracy of Logistic: %.2f' % lr.score(test_vecs, y_test)
print 'Train Accuracy of Logistic: %.2f' % lr.score(train_vecs, y_train)

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

knn = neighbors.KNeighborsClassifier(15, weights='distance')
knn.fit(train_vecs, y_train)
print 'Test Accuracy of knn: %.2f' % knn.score(test_vecs, y_test)
print 'Train Accuracy of knn: %.2f' % knn.score(train_vecs, y_train)
print metrics.classification_report(y_train, knn.predict(train_vecs))

pred_probas = knn.predict_proba(test_vecs)[:, 1]

plt.figure(3)
fpr, tpr, _ = roc_curve(y_test, pred_probas)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='area = %.2f' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc='lower right')
plt.show()