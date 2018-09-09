import jieba
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical


class LSTMClassifier(object):

	def __init__():
		pass

	def cut_text(data):
		data['content'] = data['content'].apply(lambda x: " ".join(jieba.cut(x, cut_all=False)))
		return data

	def get_matrix(data):
		data['content'] = self.cut_text(data)
		tokenizer = Tokenizer(num_words=self.max_fatures, split=' ')
	    tokenizer.fit_on_texts(data['content'].values)
	    X = tokenizer.texts_to_sequences(data['content'].values)
	    X = pad_sequences(X)
		print "define tokenizer ... "
	    self.tokenizer = tokenizer
	    return X

	def get_model():
		print "prepare model ... "
	    embed_dim = 128
	    lstm_out = 196
	    model = Sequential()
	    model.add(Embedding(self.max_fatures, embed_dim,input_length = X.shape[1]))
	    model.add(SpatialDropout1D(0.4))
	    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
	    model.add(Dense(2,activation='softmax'))
	    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
	    print(model.summary())
	    return model