from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer  # 将文本中的词语转换为词频矩阵
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB  # 伯努利朴素贝叶斯
from sklearn.naive_bayes import MultinomialNB # 多项分布朴素贝叶斯
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# np.random.seed(100)
# print("set seed : ", np.random.random())

# 数据介绍
# 训练集大小：len(newsgroups_train.data) = 2034
# 测试机大小：len(newsgroups_test.data) = 1353

### Converting text to vectors ###
### extract TF-IDF vectors ###
def method_
    cntvect = feature_extraction.text.CountVectorizer(max_features=1000)
    trainX = cntvect.fit_transform(data_train.data)
    trainY = newsgroups_train.target
    testX  = cntvect.transform(data_test.data)
    testY = newsgroups_test.target

    bmodel = naive_bayes.BernoulliNB(alpha=0.001)
    bmodel.fit(trainX, trainY)

    predY = bmodel.predict(testX)
    accuracy = metrics.accuracy_score(testY, predY)
    print(accuracy)

def method_tfidf():

    vectorizer = TfidfVectorizer()
    x_train = vectorizer.fit_transform(data_train.data)  # 学习并返回术语文档矩阵
    x_test = vectorizer.transform(data_test.data)  # 将文档转换为术语文档矩阵
    y_train, y_test = data_train.target, data_test.target

    # BernoulliNB 伯努利朴素贝叶斯实现文本分类
    clf1 = MultinomialNB(alpha=0.1)  # aplpha是平滑参数
    clf1.fit(x_train, y_train)
    pred = clf1.predict(x_test)
    accuracy = accuracy_score(y_test, pred)
    print('tfidf-accuracy: ', accuracy)


if __name__ == "__main__":
    # strip away headers/footers/quotes from the text
    removeset = ('headers', 'footers', 'quotes')

    # load only a sub-selection of the categories
    cats = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']

    # load train and test dataset
    data_train = fetch_20newsgroups(
        subset='train', categories=cats, remove=removeset, data_home='./')
    data_test = fetch_20newsgroups(
        subset='test', categories=cats, remove=removeset, data_home='./')

    method_tfidf()


    