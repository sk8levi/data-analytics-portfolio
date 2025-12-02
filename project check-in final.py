
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

data = pd.read_csv('/Users/levi/Desktop/QMSS/5067 NLP/Combined_News_DJIA.csv')

# Combine Top Headlines
top_cols = [c for c in data.columns if re.match(r'^Top\d+$', c)]
top_cols = sorted(top_cols, key=lambda x: int(x[3:]))

def strip_bytes(s):
    return str(s).lstrip("b'").lstrip('b"').strip("'\"") if pd.notna(s) else ""

for c in top_cols:
    data[c] = data[c].apply(strip_bytes)

data["daily_text_raw"] = data[top_cols].fillna("").agg(" ".join, axis=1)

def clean_txt(s):
    s = str(s)
    s = re.sub(r"\bU\.S\.A?\b", "US_TOKEN", s, flags=re.IGNORECASE)
    s = re.sub(r"[^A-Za-z']+", " ", s).lower().strip()
    s = s.replace("us_token", "U.S.")
    return s

data["daily_text"] = data["daily_text_raw"].apply(clean_txt)

stop_words = set(stopwords.words("english"))

def remove_sw(s):
    return " ".join([w for w in s.split() if w not in stop_words])

data["daily_text_nostop"] = data["daily_text"].apply(remove_sw)

lemma = WordNetLemmatizer()

def lemmatize_text(s):
    cleaned = " ".join(lemma.lemmatize(w) for w in s.split())
    cleaned = " ".join([w for w in cleaned.split() if len(w) > 1 or w in ["a", "i"]])
    return cleaned

data["daily_text_lemma"] = data["daily_text_nostop"].apply(lemmatize_text)

# Target
y = data["Label"]

vectorizers = [
    ("CountVectorizer", "2,2-gram", CountVectorizer(ngram_range=(2,2), min_df=3)),
    ("CountVectorizer", "1,2-gram", CountVectorizer(ngram_range=(1,2), min_df=3)),
    ("TF-IDF", "2,2-gram", TfidfVectorizer(ngram_range=(2,2), min_df=3)),
    ("TF-IDF", "1,2-gram", TfidfVectorizer(ngram_range=(1,2), min_df=3)),
]

rf_baseline_params = {
    "max_depth": 50,
    "n_estimators": 200,
    "min_samples_split": 2,
    "random_state": 42
}

params = {
    "max_depth": [10, 50, None],
    "n_estimators": [50, 100, 200],
    "max_features": ["sqrt", "log2"],
    "min_samples_split": [2, 5]
}

for vec_name, ngram_desc, vec in vectorizers:
    X = vec.fit_transform(data["daily_text_lemma"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    rf = RandomForestClassifier(**rf_baseline_params)
    rf.fit(X_train, y_train)
    print(vec_name, ngram_desc, "RF Test Accuracy:", rf.score(X_test, y_test))

    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    print(vec_name, ngram_desc, "NB Test Accuracy:", nb.score(X_test, y_test))

    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)
    print(vec_name, ngram_desc, "LogReg Test Accuracy:", logreg.score(X_test, y_test))

    svm = LinearSVC()
    svm.fit(X_train, y_train)
    print(vec_name, ngram_desc, "SVM Test Accuracy:", svm.score(X_test, y_test))

