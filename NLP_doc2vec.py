import gensim
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


# Do some very minor text preprocessing
def cleanText(corpus):
    punctuation = """.,?!:;(){}[]"""
    corpus = [z.lower().replace('\n', '') for z in corpus]
    corpus = [z.replace('<br />', ' ') for z in corpus]

    # treat punctuation as individual words
    for c in punctuation:
        corpus = [z.replace(c, ' ') for z in corpus]
    corpus = [z.split() for z in corpus]
    return corpus


# Gensim's Doc2Vec implementation requires each document/paragraph to have a label associated with it.
# We do this by using the LabeledSentence method. The format will be "TRAIN_i" or "TEST_i" where "i" is
# a dummy index of the review.
def labelizeReviews(reviews, label_type):
    labelized = []
    for i, v in enumerate(reviews):
        label = '%s_%s' % (label_type, i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized


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


def storeVecs(input, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(input, fw)
    fw.close()


def buildtrainvecs(model_dm, model_dbow, x_train, unsup_reviews, iteration):
    # We pass through the data set multiple times, shuffling the training reviews each time to improve accuracy.
    temp2 = x_train[:]
    temp2.extend(unsup_reviews)
    # all_train_reviews = np.concatenate((x_train))
    print "+==== Training =====+"
    for epoch in range(iteration):
        # perm = np.random.permutation(all_train_reviews.shape[0])
        print "+=== Iteration %d ===+" % epoch
        random.shuffle(temp2)
        model_dm.train(temp2)
        model_dbow.train(temp2)
    train_vecs_dm = getVecs(model_dm, x_train, size)
    train_vecs_dbow = getVecs(model_dbow, x_train, size)
    train_vecs = np.hstack((train_vecs_dm, train_vecs_dbow))
    return train_vecs


def buildtestvecs(model_dm, model_dbow, x_test, iteration):
    # train over test set
    x_test = np.array(x_test)
    temp3 = x_test[:]
    print "+===== Testing =====+"
    for epoch in range(iteration):
        print "+=== Iteration %d ===+" % epoch
        random.shuffle(temp3)
        model_dm.train(temp3)
        model_dbow.train(temp3)

    # Construct vectors for test reviews
    test_vecs_dm = getVecs(model_dm, x_test, size)
    test_vecs_dbow = getVecs(model_dbow, x_test, size)
    test_vecs = np.hstack((test_vecs_dm, test_vecs_dbow))
    return test_vecs


def ShowROC(Classifier, figure):
    pred_probas = Classifier.predict_proba(test_vecs)[:, 1]

    plt.figure(figure)
    fpr, tpr, _ = roc_curve(y_test, pred_probas)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='area = %.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc='lower right')
    plt.show()


def load_sup_set():
    pos_reviews = []
    neg_reviews = []
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
    # use 1 for positive sentiment, 0 for negative
    y = np.concatenate((np.ones(len(pos_reviews)), np.zeros(len(neg_reviews))))
    x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos_reviews, neg_reviews)), y, test_size=0.2)
    return x_train, x_test, y_train, y_test


def load_unsup_set():
    unsup_reviews1 = []
    unsup_reviews2 = []
    unsup_reviews = []
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
    unsup_reviews.extend(unsup_reviews1)
    unsup_reviews.extend(unsup_reviews2)
    return unsup_reviews

# ===========================================
# load data
x_train, x_test, y_train, y_test = load_sup_set()
unsup_reviews = load_unsup_set()
# Pretreatment
x_train = cleanText(x_train)
x_test = cleanText(x_test)
unsup_reviews = cleanText(unsup_reviews)
x_train = labelizeReviews(x_train, 'TRAIN')
x_test = labelizeReviews(x_test, 'TEST')
unsup_reviews = labelizeReviews(unsup_reviews, 'UNSUP')

# ===========================================
# Train Doc2Vec models
size = 400
iteration = 5

# instantiate our DM and DBOW models
model_dm = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, workers=3)
model_dbow = gensim.models.Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, dm=0, workers=3)

# build vocab over all reviews
temp1 = x_train[:]
temp1.extend(x_test)
temp1.extend(unsup_reviews)
model_dm.build_vocab(temp1)
model_dbow.build_vocab(temp1)

# get train_vecs
train_vecs = buildtrainvecs(model_dm, model_dbow, x_train, unsup_reviews, iteration)

# get test vecs
test_vecs = buildtestvecs(model_dm, model_dbow, x_test, iteration)


storeVecs(test_vecs, 'test_vecs.txt')
storeVecs(train_vecs, 'train_vecs.txt')
storeVecs(y_test, 'y_test.txt')
storeVecs(y_train, 'y_train.txt')

# ===========================================
# Logistic Regression
lr = SGDClassifier(loss='log', penalty='l1')
lr.fit(train_vecs, y_train)
print 'Test Accuracy of Logistic: %.2f' % lr.score(test_vecs, y_test)
print 'Train Accuracy of Logistic: %.2f' % lr.score(train_vecs, y_train)

ShowROC(lr, 1)

# SVM
svm = SVC(C=50, gamma=1, probability=True)
svm.fit(train_vecs, y_train)
print 'Test Accuracy of SVM: %.2f' % svm.score(test_vecs, y_test)
print 'Train Accuracy of SVM: %.2f' % svm.score(train_vecs, y_train)
print metrics.classification_report(y_train, svm.predict(train_vecs))

ShowROC(svm, 2)

# KNN
knn = neighbors.KNeighborsClassifier(15, weights='distance')
knn.fit(train_vecs, y_train)
print 'Test Accuracy of knn: %.2f' % knn.score(test_vecs, y_test)
print 'Train Accuracy of knn: %.2f' % knn.score(train_vecs, y_train)
print metrics.classification_report(y_train, knn.predict(train_vecs))

ShowROC(knn, 3)