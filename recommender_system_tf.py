# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 19:32:27 2019

@author: Gaurav

reommender system based on tensorflow tutorial from the blog:
https://vitobellini.github.io/posts/2018/01/03/how-to-build-a-recommender-system-in-tensorflow.html
"""

#import the libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Embedding, Input, Concatenate, Dot, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
import time
import os

from tensorflow.keras.callbacks import TensorBoard

#tf.enable_eager_execution()

#%% load data

df = pd.read_csv('ratings.csv')
df.drop(columns='timestamp', axis=1, inplace=True)

df_pivot = df.pivot('userid', 'movieid', 'rating')

#%% understand dataset
sns.countplot(x='rating', data=df)
sns.violinplot(x='rating', y='movieid', data=df)
sns.distplot(df['userid'])
sns.heatmap(df.corr(), cmap='RdYlGn_r', annot=True, fmt='g')
userid_unique = df['userid'].max()
movieid_unique = df['movieid'].max()
#%% split data into test and train data
y_categorical = to_categorical(df.iloc[:,2])
x_train, x_test, y_train, y_test = train_test_split(df.iloc[:,:2], y_categorical, test_size=0.2, random_state=42)


#%% keras model

userid_input_layer = Input(shape=(1,))
userid_embedding_layer = Embedding(userid_unique+1, 10, name='userid_embedding_layer')(userid_input_layer)
flatten_layer1 = Flatten()(userid_embedding_layer)

movieid_input_layer = Input(shape=(1,))
movieid_embedding_layer = Embedding(movieid_unique+1, 10, name='movieid_embedding_layer')(movieid_input_layer)
flatten_layer2 = Flatten()(movieid_embedding_layer)
#dot_layer = Concatenate(axis=-1)([flatten_layer1, flatten_layer2])
dot_layer = Dot(axes=1)([flatten_layer1, flatten_layer2])
output_layer = Dense(6, activation='softmax')(dot_layer)

model = Model(inputs=[userid_input_layer, movieid_input_layer], outputs=output_layer)
model.summary()

#%% compile keras model

model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['acc'])

#%% train keras model
loc = os.getcwd()
tensorboard = TensorBoard(log_dir=loc+'\logs\model_{}'.format(int(time.time())),histogram_freq=0, write_graph=True, write_images=True)
model.fit([x_train['userid'],x_train['movieid']], y_train, batch_size=10000, epochs=20, validation_split=0.2, callbacks=[tensorboard])

#%%
model.evaluate([x_test['userid'],x_test['movieid']], y_test)

predict = np.argmax(model.predict([x_test['userid'],x_test['movieid']]), axis=-1)
sns.scatterplot(predict[:3000], np.argmax(y_test[:3000], axis=-1))
