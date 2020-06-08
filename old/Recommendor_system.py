"""import libraries"""
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Embedding, Dense, Flatten, Dot
from tensorflow.keras.models import Model
import seaborn as sns
import matplotlib.pyplot as plt

#%%
"""retrieving the dataset"""
df = pd.read_csv('ratings.csv')
df = pd.DataFrame(df)

num_unique_book_id = len(df['book_id'].unique())
num_unique_user_id = len(df['user_id'].unique())
num_unique_ratings = len(df['rating'].unique())

x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 50)

#%%
"""keras model preparation"""
book_input = Input(shape=(1,), name='book_input')
book_embedding = Embedding(num_unique_book_id+1,10, name='book_embedding')(book_input)
book_flatten = Flatten(name='book_flatten')(book_embedding)

user_input = Input(shape=(1,), name='user_input')
user_embedding = Embedding(num_unique_user_id+1,10, name='user_embedding')(user_input)
user_flatten = Flatten(name='user_flatten')(user_embedding)

prod = Dot(name='prod',axes=1)([book_flatten, user_flatten])
prod_dense = Dense(1)(prod)

model = Model(inputs=[book_input, user_input],outputs=prod_dense)

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

'''model learning'''
history= model.fit(list(x_train.T),y_train,
          steps_per_epoch=20,
          epochs=100,
          validation_data=(list(x_test.T),y_test),
          validation_steps=2
          )
#%%
pred= model.predict(list(x_test.T))
model.evaluate(list(x_test.T),y_test
          )
plt.scatter(y_test, pred)
sns.violinplot(y_test, pred[:,0])

#%%
'''plot learning trend'''
plt.figure()
plt.plot(history.history['loss'], color='k')
plt.plot(history.history['val_loss'], color='b')

#%%
'''extracting embedding layer'''
book_embedding_weights= model.get_layer('book_embedding').get_weights()[0]

from sklearn.decomposition import PCA

pca= PCA(n_components=2)
pca_result= pca.fit_transform(book_embedding_weights)
plt.scatter(pca_result[:,0], pca_result[:,1], s=1)

#%%
import seaborn as sns
import pandas as pd

x= np.linspace(0,100,100)
y= np.random.randint(0,5,(100,))
df= pd.DataFrame(np.vstack((x,y)).T)
sns.relplot(x=0, y=1,data= df, kind='line')
