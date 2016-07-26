import os,time
import numpy as np
from gensim.models.word2vec import Word2Vec 
import os
import numpy as np
from sklearn.svm import SVC
from sklearn import metrics, neighbors
from sklearn.externals import joblib
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
%matplotlib inline

NUM = 800 #number of training sets

data = "/home/paul/Documents/NLP/review_polarity/txt_sentoken/"
neg_data = data+"neg/"
pos_data = data+"pos/"
neg_files = os.listdir(neg_data)
pos_files = os.listdir(pos_data)

#load GoogleNews-vectors
def loadGoogleVector():
	t1 = time.clock()
	vector_bin = "/home/paul/Data/GoogleNews-vectors-negative300.bin"
	vector_bin2 = "/home/paul/Data/"
	model = Word2Vec.load_word2vec_format(vector_bin, binary=True)
	t2 = time.clock()
	print ("loading GoogleVector time : %.2f" % (t2 - t1))
	return model

# preprocess
def cleanText(corpus):
    punctuation = """.,?!:;(){}[]"""
    corpus = [z.lower().replace('\n', '') for z in corpus]
    corpus = [z.replace('<br />', ' ') for z in corpus]
    # treat punctuation as ''
    for c in punctuation:
        corpus = [z.replace(c, '') for z in corpus]
    corpus = [z.split() for z in corpus]
    return corpus
# buidArticleVector
def buildArticleVector(text,model,size=300):
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
#load traing sets and labels
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
#    load testing sets and labels
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
    y_test = np.concatenate(([0 for i in range((1000-NUM))],[1 for i in range((1000-NUM))]))
    return articles_test,y_test




# training
def Train_SVM(vecs_train,y_train):
	svm = SVC(C=50, gamma=1, probability=True)
	svm.fit(vecs_train,y_train)
	print 'Train Accuracy of SVM: %.2f' % svm.score(vecs_train, y_train)
	print metrics.classification_report(y_train, svm.predict(vecs_train))
	# store parameters of model
	os.chdir("SVM_MODEL/")
	joblib.dump(svm, "train_model.m")

def Test_SVM(vecs_test, y_test)
	#testing
	svm = joblib.load("train_model.m")
	print 'Test Accuracy of SVM: %.2f' % svm.score(vecs_test, y_test)
	print metrics.classification_report(y_test, svm.predict(vecs_test))

def ROC(vecs_test,y_test)
	pred_probas = svm.predict_proba(vecs_test)[:,1]
	fpr, tpr, _ = roc_curve(y_test, pred_probas)
	roc_auc = auc(fpr, tpr)
	plt.plot(fpr, tpr, label='area = %.2f' % roc_auc)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.legend(loc='lower right')
	plt.show()

model = loadGoogleVector()
t1 = time.time()
articles_train,y_train = loadTrainData()
articles_test,y_test = loadTestData()
vecs_train = np.concatenate([buildArticleVector(article,model) for article in articles_train])
vecs_test = np.concatenate([buildArticleVector(article,model) for article in articles_test])
t2 = time.time()
print "getVector time: ",t2 - t1
Train_SVM(vecs_train,y_train)
Test_SVM(vecs_test, y_test)
ROC(vecs_test,y_test)	