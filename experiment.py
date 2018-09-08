
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
import jieba


# In[2]:


data = pd.read_csv('ai_challenger_sentiment_analysis_trainingset_20180816/sentiment_analysis_trainingset.csv')


# In[36]:


max_fatures = 2000


# In[30]:


def append_label(data):
    data = data[['content','others_overall_experience']]
    data = data[data.others_overall_experience != 0]
    # data['content'] = data['content'].apply(lambda x: x.lower())
    # data['content'] = data['content'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

    label = ['p' if x>0 else 'n' for x in data.others_overall_experience.values]
    data['label'] = label

    print(data[ data['label'] == 'p'].size)
    print(data[ data['label'] == 'n'].size)
    return data


# In[33]:


def get_matrix(data):

    data['content'] = data['content'].apply(lambda x: " ".join(jieba.cut(x, cut_all=False)))

    for idx,row in data.iterrows():
        row[0] = row[0].replace('rt',' ').replace(u'～', ' ').replace(u'，', ' ').replace(u'！', ' ')

    tokenizer = Tokenizer(num_words=max_fatures, split=' ')
    tokenizer.fit_on_texts(data['content'].values)
    X = tokenizer.texts_to_sequences(data['content'].values)
    X = pad_sequences(X)

    print X.shape
    return X


# In[32]:


def get_model():
    embed_dim = 128
    lstm_out = 196

    model = Sequential()
    model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    print(model.summary())
    return model


# In[34]:


data = append_label(data)
X = get_matrix(data)
model = get_model()


# In[ ]:


Y = pd.get_dummies(data['label']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


# In[22]:


Y = pd.get_dummies(data['label']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


# In[ ]:


batch_size = 32
model.fit(X_train, Y_train, epochs = 7, batch_size=batch_size, verbose = 2)


# In[ ]:


validation_size = 1500

X_validate = X_test[-validation_size:]
Y_validate = Y_test[-validation_size:]
X_test = X_test[:-validation_size]
Y_test = Y_test[:-validation_size]
score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))


# In[ ]:


pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0
for x in range(len(X_validate)):
    
    result = model.predict(X_validate[x].reshape(1,X_test.shape[1]),batch_size=1,verbose = 2)[0]
   
    if np.argmax(result) == np.argmax(Y_validate[x]):
        if np.argmax(Y_validate[x]) == 0:
            neg_correct += 1
        else:
            pos_correct += 1
       
    if np.argmax(Y_validate[x]) == 0:
        neg_cnt += 1
    else:
        pos_cnt += 1



print("pos_acc", pos_correct/pos_cnt*100, "%")
print("neg_acc", neg_correct/neg_cnt*100, "%")

with open('results.txt', 'w') as f:
    f.write('\n'.join(result)) 

