{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Head**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 131,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import pandas_profiling\n",
    "import re\n",
    "import sklearn\n",
    "import time\n",
    "from sklearn.feature_extraction.text import CountVectorizer\n",
    "from sklearn.feature_extraction.text import TfidfTransformer\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn import tree\n",
    "from sklearn import metrics\n",
    "from sklearn.naive_bayes import BernoulliNB\n",
    "from sklearn.naive_bayes import MultinomialNB\n",
    "from sklearn.naive_bayes import ComplementNB\n",
    "from sklearn.linear_model import LogisticRegression"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Cleaning**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 132,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('trainset.csv', delimiter=';')\n",
    "df.product_name = df.product_name.astype(str)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 133,
   "metadata": {},
   "outputs": [],
   "source": [
    "df['clean_product_name'] = df['product_name'].str.lower()\n",
    "df['clean_product_name'] = df['clean_product_name'].str.replace(r'[,:\\(\\)\"_|#№]', u'')\n",
    "df['clean_product_name'] = df['clean_product_name'].str.replace(r'сиг([и]?|ар?(ет|илл?)[ыа])[а-я]*', u'')\n",
    "df['clean_product_name'] = df['clean_product_name'].str.replace(r'мрц\\s?\\d*[,.-=-]*\\d*\\s?', u'')\n",
    "# df['clean_product_name'] = df['clean_product_name'].str.replace(r'(?<=[^0-9])[\\_\\*\\'\\\"\\.,]', u'')\n",
    "# df['clean_product_name'] = df['clean_product_name'].str.replace(r'\\d?\\d?\\s?\\(?ш[т\\.]у?к?\\)?', u'')\n",
    "# df['clean_product_name'] = df['clean_product_name'].str.replace(r'([^а-я]*р[.у][^а-я]б?|р[.у]б?л?е?й?$)', u'')\n",
    "# df['clean_product_name'] = df['clean_product_name'].str.replace(r'\\d\\d?\\d?[-\\/,]\\d?\\d?\\d?', u'')\n",
    "# df['clean_product_name'] = df['clean_product_name'].str.replace(r'^(аа|яя)\\W?', u'')\n",
    "# df['clean_product_name'] = df['clean_product_name'].str.replace(r'\\(\\s?\\d*\\s?[,\\.=-]?\\s?\\d*\\s?\\)', u'')\n",
    "# df['clean_product_name'] = df['clean_product_name'].str.replace(r'(б[.л]о?к?$|п[а.]ч?к?а?[^а-я])', u'')\n",
    "# df['clean_product_name'] = df['clean_product_name'].str.replace(r'(^\\d{2,}\\s*|[^а-я]\\d{3,}\\s*$)', u'')\n",
    "# df['clean_product_name'] = df['clean_product_name'].str.replace(r'(\\[\\d*|\\d*\\])', u'')\n",
    "df['clean_product_name'] = df['clean_product_name'].str.replace(r'\\/ш\\/', u'')\n",
    "df['clean_product_name'] = df['clean_product_name'].str.replace(r'[\\;\\!\\\\~\\+\\*]', u'')\n",
    "df['clean_product_name'] = df['clean_product_name'].str.replace(r'(\\/а|п$)', u'')\n",
    "df['clean_product_name'] = df['clean_product_name'].str.replace(r'пач[.к]?а?', u'')\n",
    "df['clean_product_name'] = df['clean_product_name'].str.replace(r'\\d*\\s?МГ', u'')\n",
    "df['clean_product_name'] = df['clean_product_name'].str.replace(r'папиро?с?ы?', u'')\n",
    "# df['clean_product_name'] = df['clean_product_name'].str.replace(r'\\d{1,}р\\s?[^а-я]', u'')\n",
    "df['clean_product_name'] = df['clean_product_name'].str.replace(r'\\d{4,}', u'')\n",
    "df['clean_product_name'] = df['clean_product_name'].str.replace(r'р\\.', u'')\n",
    "df['clean_product_name'] = df['clean_product_name'].str.replace(r'-00', u'')\n",
    "# df['clean_product_name'] = df['clean_product_name'].str.replace(r'(^ | $)', u'')\n",
    "df['clean_product_name'] = df['clean_product_name'].str.upper()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "metadata": {},
   "outputs": [],
   "source": [
    "# df.sample(100, random_state=65663)\n",
    "# for i in df.sample(100).clean_product_name:\n",
    "#     print(i)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**CountVectorizer & TF-IDF**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 134,
   "metadata": {},
   "outputs": [],
   "source": [
    "vectorizer = CountVectorizer()\n",
    "transformer = TfidfTransformer()\n",
    "X = df['clean_product_name']\n",
    "y = df['brand_variant_code']\n",
    "y_category = y.astype('category').cat.codes\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y_category)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 135,
   "metadata": {},
   "outputs": [],
   "source": [
    "matrix_train = vectorizer.fit_transform(X_train)\n",
    "tf_idf_train = transformer.fit_transform(matrix_train)\n",
    "matrix_test = vectorizer.transform(X_test)\n",
    "tf_idf_test = transformer.transform(matrix_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Algorithm I - Decision Tree**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 136,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "CPU times: user 40.4 s, sys: 344 ms, total: 40.8 s\n",
      "Wall time: 40.8 s\n"
     ]
    }
   ],
   "source": [
    "%%time\n",
    "clf = tree.DecisionTreeClassifier(splitter='random')\n",
    "clf.fit(tf_idf_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 137,
   "metadata": {},
   "outputs": [],
   "source": [
    "predicted = clf.predict(tf_idf_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Accuracy**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 138,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9037390480307237"
      ]
     },
     "execution_count": 138,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sklearn.metrics.accuracy_score(y_test, predicted)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Algorithm II - Naive Bayes**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 125,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "CPU times: user 2.37 s, sys: 1.04 s, total: 3.41 s\n",
      "Wall time: 3.4 s\n"
     ]
    }
   ],
   "source": [
    "%%time\n",
    "nb = MultinomialNB(alpha=0.01, fit_prior='False')\n",
    "nb.fit(tf_idf_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 126,
   "metadata": {},
   "outputs": [],
   "source": [
    "predicted = nb.predict(tf_idf_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Accuracy**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 127,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "CPU times: user 8.23 ms, sys: 1.1 ms, total: 9.33 ms\n",
      "Wall time: 7.24 ms\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "0.8475618879781875"
      ]
     },
     "execution_count": 127,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "%%time\n",
    "sklearn.metrics.accuracy_score(y_test, predicted)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Algorithm III - Logistic Regression**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 128,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "CPU times: user 38min 54s, sys: 56.4 ms, total: 38min 54s\n",
      "Wall time: 38min 54s\n"
     ]
    }
   ],
   "source": [
    "%%time\n",
    "lr = LogisticRegression(random_state=0, solver='liblinear', multi_class='auto').fit(tf_idf_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 129,
   "metadata": {},
   "outputs": [],
   "source": [
    "predicted = lr.predict(tf_idf_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Accuracy**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 130,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "CPU times: user 8.52 ms, sys: 2.11 ms, total: 10.6 ms\n",
      "Wall time: 8.95 ms\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "0.8636552561139836"
      ]
     },
     "execution_count": 130,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "%%time\n",
    "sklearn.metrics.accuracy_score(y_test, predicted)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
