import jieba
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical


class LSTMClassifier(object):

	def __init__(self):
		self.batch_size = 32
		self.max_fatures = 2000

	def cut_text(self, data):
		data['content'] = data['content'].apply(lambda x: " ".join(jieba.cut(x, cut_all=False)))
		return data

	def get_matrix(self, data):
		data['content'] = self.cut_text(data)
		tokenizer = Tokenizer(num_words=self.max_fatures, split=' ')
		tokenizer.fit_on_texts(data['content'].values)
		X = tokenizer.texts_to_sequences(data['content'].values)
		X = pad_sequences(X)
		print "define tokenizer ... "
		self.tokenizer = tokenizer
		return X

	def get_model(self, input_size):
		print "prepare model ... "
		embed_dim = 128
		lstm_out = 196
		model = Sequential()
		model.add(Embedding(self.max_fatures, embed_dim, input_length=input_size))
		model.add(SpatialDropout1D(0.4))
		model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
		model.add(Dense(2,activation='softmax'))
		model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
		print(model.summary())
		return model

	def fit_model(self, model, X_train, Y_train):
		return model.fit(X_train, Y_train, epochs = 7, batch_size=self.batch_size, verbose = 2)