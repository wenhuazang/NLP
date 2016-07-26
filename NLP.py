
# coding: utf-8

# # loading googleNews

# In[1]:

import os,time
import numpy as np
from gensim.models.word2vec import Word2Vec 
t1 = time.clock()
vector_bin = "/home/paul/Data/GoogleNews-vectors-negative300.bin"
vector_bin2 = "/home/paul/Data/"
model = Word2Vec.load_word2vec_format(vector_bin, binary=True)
#print model['computer']
t2 = time.clock()
print ("elapsed time : %.2f" % (t2 - t1))


## #　＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝　＃

# In[3]:

import os
import numpy as np
NUM = 800 #training sets

data = "/home/paul/Documents/NLP/review_polarity/txt_sentoken/"
neg_data = data+"neg/"
pos_data = data+"pos/"
neg_files = os.listdir(neg_data)
pos_files = os.listdir(pos_data)


# # preprocess

# In[4]:

def cleanText(corpus):
    punctuation = """.,?!:;(){}[]"""
    corpus = [z.lower().replace('\n', '') for z in corpus]
    corpus = [z.replace('<br />', ' ') for z in corpus]
    # treat punctuation as ''
    for c in punctuation:
        corpus = [z.replace(c, '') for z in corpus]
    corpus = [z.split() for z in corpus]
    return corpus


# # buildArticleVecs

# In[5]:

def buildArticleVector(text, size=300):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += model[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


# # load traing sets and labels

# In[6]:

def loadTrainData():
    articles_train = []
    for i in range(NUM):
        with open(neg_data + neg_files[i],"r") as infile:
            neg_article = infile.readlines()
            article = cleanText(neg_article)
            article = np.concatenate(article)
            articles_train.append(article)
    for i in range(NUM):        
        with open(pos_data + pos_files[i],"r") as infile:
            pos_article = infile.readlines()
            article = cleanText(pos_article)
            article = np.concatenate(article)
            articles_train.append(article)
    #articles = np.concatenate(articles)
    # given labels
    y_train = np.concatenate(([0 for i in range(NUM)],[1 for i in range(NUM)]))
    return articles_train,y_train


# # load testing sets and labels

# In[7]:

def loadTestData():
    articles_test = []
    for i in range(NUM,1000):
        with open(neg_data + neg_files[i],"r") as infile:
            neg_article = infile.readlines()
            article = cleanText(neg_article)
            article = np.concatenate(article)
            articles_test.append(article)
    for i in range(NUM,1000):        
        with open(pos_data + pos_files[i],"r") as infile:
            pos_article = infile.readlines()
            article = cleanText(pos_article)
            article = np.concatenate(article)
            articles_test.append(article)
    #articles = np.concatenate(articles)
    # given labels
    y_test = np.concatenate(([0 for i in range((1000-NUM))],[1 for i in range((1000-NUM))]))
    return articles_test,y_test


# In[9]:

t1 = time.time()
articles_train,y_train = loadTrainData()
articles_test,y_test = loadTestData()
vecs_train = np.concatenate([buildArticleVector(article) for article in articles_train])
vecs_test = np.concatenate([buildArticleVector(article) for article in articles_test])
t2 = time.time()
print "elapsed time: ",t2 - t1


# In[10]:

len(vecs_train),len(vecs_test)


# # training

# In[13]:

from sklearn.svm import SVC
from sklearn import metrics, neighbors
svm = SVC(C=50, gamma=1, probability=True)
#svm = SVC(probability=True)
svm.fit(vecs_train,y_train)
#print 'Test Accuracy of SVM: %.2f' % svm.score(test_vecs, y_test)
print 'Train Accuracy of SVM: %.2f' % svm.score(vecs_train, y_train)
print metrics.classification_report(y_train, svm.predict(vecs_train))


# # store parameters of model

# In[22]:

from sklearn.externals import joblib
os.chdir("SVM_MODEL/")
joblib.dump(svm, "train_model.m")


# # testing

# In[27]:

svm = joblib.load("train_model.m")
print 'Test Accuracy of SVM: %.2f' % svm.score(vecs_test, y_test)
print metrics.classification_report(y_test, svm.predict(vecs_test))


# # predict single samples

# In[19]:

svm.predict([vecs_test[311]])


# # ROC curve

# In[49]:

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
pred_probas = svm.predict_proba(vecs_test)[:,1]

fpr, tpr, _ = roc_curve(y_test, pred_probas)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='area = %.2f' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc='lower right')
plt.show()


# In[44]:

svm.predict_proba(vecs_test)[:,1]


# In[47]:

svm.predict_proba(vecs_test)


# In[52]:

article


# In[ ]:



