# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 21:16:31 2020

@author: prana
"""

import numpy as np
import pandas as pd

dt = pd.read_csv("Restaurant_Reviews.tsv",delimiter="\t")
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
data = []
for i in range(0,1000):
    review = dt["Review"][i]
    review = re.sub('[^a-zA-Z]', ' ',review)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    data.append(review)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2000)
x = cv.fit_transform(data).toarray()


y = dt.iloc[:,1:2].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(units = 1565 ,init = "random_uniform",activation = "relu",batch_size=32))
model.add(Dense(units = 3000 ,init = "random_uniform",activation = "relu",batch_size=32))

model.add(Dense(units = 1 ,init = "random_uniform",activation = "sigmoid",batch_size=32))
model.compile(optimizer = "adam",loss = "binary_crossentropy",metrics = ["accuracy"])
model.fit(x_train,y_train,epochs  = 100)
x_train.shape

y_pred = model.predict(x_test)

y_pred = (y_pred >0.5)