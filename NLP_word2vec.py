from sklearn import metrics, neighbors
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import SGDClassifier
from gensim.models.word2vec import Word2Vec
from sklearn.preprocessing import scale

import numpy as np
import os
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.svm import SVC

data_path = "review_polarity/txt_sentoken/"
pos_files = os.listdir(data_path + "pos/")
neg_files = os.listdir(data_path + "neg/")
pos_reviews = []
neg_reviews = []

for pos_file in pos_files:
    with open(data_path + "pos/" + pos_file, 'r') as infile:
        each = ''
        each_article = infile.readlines()
        for each_sentence in each_article:
            each += each_sentence
    pos_reviews.extend([each])
for neg_file in neg_files:
    with open(data_path + "neg/" + neg_file, 'r') as infile:
        each = ''
        each_article = infile.readlines()
        for each_sentence in each_article:
            each += each_sentence
    neg_reviews.extend([each])

# use 1 for positive sentiment, 0 for negative
y = np.concatenate((np.ones(len(pos_reviews)), np.zeros(len(neg_reviews))))

x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos_reviews, neg_reviews)), y, test_size=0.2)


# Do some very minor text preprocessing
def cleanText(corpus):
    punctuation = """.,?!:;(){}[]"""
    corpus = [z.lower().replace('\n', '') for z in corpus]
    corpus = [z.replace('<br />', ' ') for z in corpus]

    # treat punctuation as individual words
    for c in punctuation:
        corpus = [z.replace(c, ' %s ' % c) for z in corpus]
    corpus = [z.split() for z in corpus]
    return corpus


x_train = cleanText(x_train)
x_test = cleanText(x_test)

n_dim = 300
# Initialize model and build vocab
imdb_w2v = Word2Vec(size=n_dim, min_count=10)
imdb_w2v.build_vocab(x_train)

# Train the model over train_reviews (this may take several minutes)
imdb_w2v.train(x_train, epochs=10, total_examples=2000)


# Build word vector for training set by using the average value of all word vectors in the tweet, then scale
def buildWordVector(text, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += imdb_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


train_vecs = np.concatenate([buildWordVector(z, n_dim) for z in x_train])
train_vecs = scale(train_vecs)

# Train word2vec on test tweets
imdb_w2v.train(x_test, epochs=10, total_examples=2000)

test_vecs = np.concatenate([buildWordVector(z, n_dim) for z in x_test])
test_vecs = scale(test_vecs)

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

svm = SVC(C=30, gamma=0.125, probability=True)
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