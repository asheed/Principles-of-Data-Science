#!/usr/bin/env python
# -*- coding: utf8 -*

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import html5lib
from sklearn.feature_extraction.text import CountVectorizer

# data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)
# print(data.head())
# sns.pairplot(data, x_vars=['TV', 'radio', 'newspaper'], y_vars='sales', height=4.5, aspect=0.7, palette='pastel', kind='reg')
#sns.lmplot(data, x=['TV', 'radio', 'newspaper'], y='sales')
# plt.show()

texts = []

for index in range(0, 1000, 10):    # indeed에서 100개의 페이지를 살펴본다.
	page = 'http://indeed.com/jobs?q=data+scientist&start='+str(index)

	web_result = requests.get(page).text

	bs_result = BeautifulSoup(web_result, 'html5lib')

	for listing in bs_result.findAll('span', {'class':'summary'}):
		texts.append(listing.text)

print(type(texts))
print(texts[0].strip())

vect = CountVectorizer(ngram_range=(1,2), stop_words='english')

matrix = vect.fit_transform(texts)

print(len(vect.get_feature_names()))

freqs = [(word, matrix.getcol(idx).sum()) for word, idx in vect.vocabulary_.items()]
#sort from largest to smallest
for phrase, times in sorted(freqs, key = lambda x: -x[1])[:25]:
    print(phrase, times)
