import pandas as pd
import numpy as np
import pandas_profiling
import re
import sklearn
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('trainset.csv', delimiter=';')
df.product_name = df.product_name.astype(str)

df['clean_product_name'] = df['product_name'].str.lower()
df['clean_product_name'] = df['clean_product_name'].str.replace(r'[,:\(\)"_|#№]', u'')
df['clean_product_name'] = df['clean_product_name'].str.replace(r'сиг([и]?|ар?(ет|илл?)[ыа])[а-я]*', u'')
df['clean_product_name'] = df['clean_product_name'].str.replace(r'мрц\s?\d*[,.-=-]*\d*\s?', u'')
# df['clean_product_name'] = df['clean_product_name'].str.replace(r'(?<=[^0-9])[\_\*\'\"\.,]', u'')
# df['clean_product_name'] = df['clean_product_name'].str.replace(r'\d?\d?\s?\(?ш[т\.]у?к?\)?', u'')
# df['clean_product_name'] = df['clean_product_name'].str.replace(r'([^а-я]*р[.у][^а-я]б?|р[.у]б?л?е?й?$)', u'')
# df['clean_product_name'] = df['clean_product_name'].str.replace(r'\d\d?\d?[-\/,]\d?\d?\d?', u'')
# df['clean_product_name'] = df['clean_product_name'].str.replace(r'^(аа|яя)\W?', u'')
# df['clean_product_name'] = df['clean_product_name'].str.replace(r'\(\s?\d*\s?[,\.=-]?\s?\d*\s?\)', u'')
# df['clean_product_name'] = df['clean_product_name'].str.replace(r'(б[.л]о?к?$|п[а.]ч?к?а?[^а-я])', u'')
# df['clean_product_name'] = df['clean_product_name'].str.replace(r'(^\d{2,}\s*|[^а-я]\d{3,}\s*$)', u'')
# df['clean_product_name'] = df['clean_product_name'].str.replace(r'(\[\d*|\d*\])', u'')
df['clean_product_name'] = df['clean_product_name'].str.replace(r'\/ш\/', u'')
df['clean_product_name'] = df['clean_product_name'].str.replace(r'[\;\!\\~\+\*]', u'')
df['clean_product_name'] = df['clean_product_name'].str.replace(r'(\/а|п$)', u'')
df['clean_product_name'] = df['clean_product_name'].str.replace(r'пач[.к]?а?', u'')
df['clean_product_name'] = df['clean_product_name'].str.replace(r'\d*\s?МГ', u'')
df['clean_product_name'] = df['clean_product_name'].str.replace(r'папиро?с?ы?', u'')
# df['clean_product_name'] = df['clean_product_name'].str.replace(r'\d{1,}р\s?[^а-я]', u'')
df['clean_product_name'] = df['clean_product_name'].str.replace(r'\d{4,}', u'')
df['clean_product_name'] = df['clean_product_name'].str.replace(r'р\.', u'')
df['clean_product_name'] = df['clean_product_name'].str.replace(r'-00', u'')
# df['clean_product_name'] = df['clean_product_name'].str.replace(r'(^ | $)', u'')
df['clean_product_name'] = df['clean_product_name'].str.upper()

vectorizer = CountVectorizer()
transformer = TfidfTransformer()
X = df['clean_product_name']
y = df['brand_variant_code']
y_category = y.astype('category').cat.codes
X_train, X_test, y_train, y_test = train_test_split(X, y_category)

matrix_train = vectorizer.fit_transform(X_train)
tf_idf_train = transformer.fit_transform(matrix_train)
matrix_test = vectorizer.transform(X_test)
tf_idf_test = transformer.transform(matrix_test)

# Algorithm I - Decision Tree
%%time
clf = tree.DecisionTreeClassifier(splitter='random')
clf.fit(tf_idf_train, y_train)

predicted = clf.predict(tf_idf_test)

sklearn.metrics.accuracy_score(y_test, predicted) # Accuracy

#############################################

# Algorithm II - Naive Bayes
%%time
nb = MultinomialNB(alpha=0.01, fit_prior='False')
nb.fit(tf_idf_train, y_train)

predicted = nb.predict(tf_idf_test)

%%time
sklearn.metrics.accuracy_score(y_test, predicted) # Accuracy

#############################################

# Algorithm III - Logistic Regression
%%time
lr = LogisticRegression(random_state=0, solver='liblinear', multi_class='auto').fit(tf_idf_train, y_train)

predicted = lr.predict(tf_idf_test)

%%time
sklearn.metrics.accuracy_score(y_test, predicted) # Accuracy
