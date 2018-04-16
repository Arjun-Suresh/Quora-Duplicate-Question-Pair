import pandas as pd
import numpy as np
import nltk
from collections import Counter
from nltk.corpus import stopwords
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cross_validation import train_test_split
import difflib
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
#import xgboost as xgb

stops = set(stopwords.words("english"))
train = pd.read_csv('Input/train.csv')[:10000]
test = pd.read_csv('Input/test.csv')[:10000]
tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))

tfidf_txt = pd.Series(train['question1'].tolist() + train['question2'].tolist() + test['question1'].tolist() + test['question2'].tolist()).astype(str)
tfidf.fit_transform(tfidf_txt)


def diff_ratios(st1, st2):
    seq = difflib.SequenceMatcher()
    seq.set_seqs(str(st1).lower(), str(st2).lower())
    return seq.ratio()


def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R



def tfidf_word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        return 0
    q1_tfidf = tfidf.transform([" ".join(q1words.keys())])
    q2_tfidf = tfidf.transform([" ".join(q2words.keys())])
    inter = np.intersect1d(q1_tfidf.indices, q2_tfidf.indices)
    shared_weights = 0
    for word_index in inter:
        shared_weights += (q1_tfidf[0, word_index] + q2_tfidf[0, word_index])
    total_weights = q1_tfidf.sum() + q2_tfidf.sum()
    return np.sum(shared_weights) / np.sum(total_weights)



def get_features(df_features):
    print('nouns...')
    df_features['question1_nouns'] = df_features.question1.map(lambda x: [w for w, t in nltk.pos_tag(nltk.word_tokenize(str(x).lower())) if t[:1] in ['N']])
    df_features['question2_nouns'] = df_features.question2.map(lambda x: [w for w, t in nltk.pos_tag(nltk.word_tokenize(str(x).lower())) if t[:1] in ['N']])
    df_features['z_noun_match'] = df_features.apply(lambda r: sum([1 for w in r.question1_nouns if w in r.question2_nouns]), axis=1)  #takes long
    print('lengths...')
    df_features['z_len1'] = df_features.question1.map(lambda x: len(str(x)))
    df_features['z_len2'] = df_features.question2.map(lambda x: len(str(x)))
    df_features['z_word_len1'] = df_features.question1.map(lambda x: len(str(x).split()))
    df_features['z_word_len2'] = df_features.question2.map(lambda x: len(str(x).split()))
    print('difflib...')
    df_features['z_match_ratio'] = df_features.apply(lambda r: diff_ratios(r.question1, r.question2), axis=1)  #takes long
    print('word match...')
    df_features['z_word_match'] = df_features.apply(word_match_share, axis=1, raw=True)
    print('tfidf...')
    df_features['z_tfidf_sum1'] = df_features.question1.map(lambda x: np.sum(tfidf.transform([str(x)]).data))
    df_features['z_tfidf_sum2'] = df_features.question2.map(lambda x: np.sum(tfidf.transform([str(x)]).data))
    df_features['z_tfidf_mean1'] = df_features.question1.map(lambda x: np.mean(tfidf.transform([str(x)]).data))
    df_features['z_tfidf_mean2'] = df_features.question2.map(lambda x: np.mean(tfidf.transform([str(x)]).data))
    df_features['z_tfidf_len1'] = df_features.question1.map(lambda x: len(tfidf.transform([str(x)]).data))
    df_features['z_tfidf_len2'] = df_features.question2.map(lambda x: len(tfidf.transform([str(x)]).data))
    #df_features['z_tfidf_share'] = df_features.apply(tfidf_word_match_share, axis=1, raw=True)
    return df_features.fillna(0.0)


train = get_features(train)
col = [c for c in train.columns if c[:1]=='z']

pos_train = train[train['is_duplicate'] == 1]
neg_train = train[train['is_duplicate'] == 0]
p = 0.165
scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
while scale > 1:
    neg_train = pd.concat([neg_train, neg_train])
    scale -=1
neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
train = pd.concat([pos_train, neg_train])

def naive_bayes(x_train, y_train, x_valid, y_valid):
    clf=MultinomialNB()
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_valid)
    k=0
    for i in range(len(y_valid)):
        if y_valid.data[i] == y_pred[i]:
            k+=1
    retutn (float(k)/float(len(y_valid)))

def xgboost(x_train, y_train, x_valid, y_valid):
    params = {}

    params["objective"] = "binary:logistic"
    params['eval_metric'] = 'error'
    params["eta"] = 0.02
    params["subsample"] = 0.7
    params["min_child_weight"] = 1
    params["colsample_bytree"] = 0.7
    params["max_depth"] = 4
    params["silent"] = 1
    params["seed"] = 1632

    d_train = xgb.DMatrix(x_train, label=y_train)
    d_valid = xgb.DMatrix(x_valid, label=y_valid)
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    bst = xgb.train(params, d_train, 500, watchlist, early_stopping_rounds=50, verbose_eval=100)
    sub = pd.DataFrame()
    sub['is_duplicate'] = bst.predict(d_valid)
    k = 0
    for i in range(len(y_valid)):
        if y_valid.data[i] == sub["is_duplicate"][i]:
            k += 1
    retutn(float(k) / float(len(y_valid)))

def svm_model(x_train, y_train, x_valid, y_valid):
    clf = svm.SVC(C=1000)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_valid)
    k = 0
    for i in range(len(y_valid)):
        if y_valid.data[i] == y_pred[i]:
            k += 1
    return(float(k) / float(len(y_valid)))


x_train, x_valid, y_train, y_valid = train_test_split(train[col], train['is_duplicate'], test_size=0.4, random_state=0)
#accuracy = naive_bayes(x_train, y_train, x_valid, y_valid)
accuracy = svm_model(x_train, y_train, x_valid, y_valid)
print (accuracy)