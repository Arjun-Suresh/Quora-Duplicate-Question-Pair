import pandas as pd
import numpy as np
import nltk
from collections import Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import difflib


class feature_extracter:

    def __init__(self, trainFile):
        self.stops = set(stopwords.words("english"))
        self.tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 1))
        self.train = pd.read_csv('Input/train.csv')
        self.tfidf_txt = pd.Series(
            self.train['question1'].tolist() + self.train['question2'].tolist()).astype(str)
        self.tfidf.fit_transform(self.tfidf_txt)


    def diff_ratios(self, st1, st2):
        seq = difflib.SequenceMatcher()
        seq.set_seqs(str(st1).lower(), str(st2).lower())
        return seq.ratio()


    def word_match_share(self, row):
        q1words = {}
        q2words = {}
        for word in str(row['question1']).lower().split():
            if word not in self.stops:
                q1words[word] = 1
        for word in str(row['question2']).lower().split():
            if word not in self.stops:
                q2words[word] = 1
        if len(q1words) == 0 or len(q2words) == 0:
            return 0
        shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
        shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
        R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
        return R



    def tfidf_word_match_share(self, row):
        q1words = {}
        q2words = {}
        for word in str(row['question1']).lower().split():
            if word not in self.stops:
                q1words[word] = 1
        for word in str(row['question2']).lower().split():
            if word not in self.stops:
                q2words[word] = 1
        if len(q1words) == 0 or len(q2words) == 0:
            return 0
        q1_tfidf = self.tfidf.transform([" ".join(q1words.keys())])
        q2_tfidf = self.tfidf.transform([" ".join(q2words.keys())])
        inter = np.intersect1d(q1_tfidf.indices, q2_tfidf.indices)
        shared_weights = 0
        for word_index in inter:
            shared_weights += (q1_tfidf[0, word_index] + q2_tfidf[0, word_index])
        total_weights = q1_tfidf.sum() + q2_tfidf.sum()
        return np.sum(shared_weights) / np.sum(total_weights)



    def get_features(self, df_features):
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
        df_features['z_match_ratio'] = df_features.apply(lambda r: self.diff_ratios(r.question1, r.question2), axis=1)  #takes long
        print('word match...')
        df_features['z_word_match'] = df_features.apply(self.word_match_share, axis=1, raw=True)
        print('tfidf...')
        df_features['z_tfidf_sum1'] = df_features.question1.map(lambda x: np.sum(self.tfidf.transform([str(x)]).data))
        df_features['z_tfidf_sum2'] = df_features.question2.map(lambda x: np.sum(self.tfidf.transform([str(x)]).data))
        df_features['z_tfidf_mean1'] = df_features.question1.map(lambda x: np.mean(self.tfidf.transform([str(x)]).data))
        df_features['z_tfidf_mean2'] = df_features.question2.map(lambda x: np.mean(self.tfidf.transform([str(x)]).data))
        df_features['z_tfidf_len1'] = df_features.question1.map(lambda x: len(self.tfidf.transform([str(x)]).data))
        df_features['z_tfidf_len2'] = df_features.question2.map(lambda x: len(self.tfidf.transform([str(x)]).data))
        df_features['z_tfidf_share'] = df_features.apply(self.tfidf_word_match_share, axis=1, raw=True)
        return df_features.fillna(0.0)