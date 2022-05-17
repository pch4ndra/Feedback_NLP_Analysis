# Library Imoprts
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
import nltk
import re
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Dropout
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import pickle

nltk.download("stopwords")
nltk.download("punkt")

# Data Import
def load_dataset(filename):
  df = pd.read_csv(filename, encoding = "latin1", names = ["Comment", "Intent"])
  df.dropna(subset = ["Comment"], inplace=True)
  df.dropna(subset = ["Intent"], inplace=True)
  print(df.head())
  intent = df["Intent"]
  unique_intent = list(set(intent))
  comments = list(df["Comment"])

  return (intent, unique_intent, comments)

intent, unique_intent, comments = load_dataset("data/train.csv")

print(comments[:5])

# Data PreProcessing
# define stemmer
stemmer = LancasterStemmer()

def cleaning(comments):
  words = []
  for c in comments:
    clean = re.sub(r'[^ a-z A-Z 0-9]', " ", c)
    w = word_tokenize(clean)
    #stemming
    words.append([i.lower() for i in w])

  return words

cleaned_words = cleaning(comments)
print(len(cleaned_words))
print(cleaned_words[:2])

# Encoding
def create_tokenizer(words, filters = '#"%&()*+,-/:;<=>@[\]^`{|}~'):
  token = Tokenizer(filters = filters)
  token.fit_on_texts(words)
  return token

def max_length(words):
  return(len(max(words, key = len)))

word_tokenizer = create_tokenizer(cleaned_words)
vocab_size = len(word_tokenizer.word_index) + 1
max_length = max_length(cleaned_words)

print("Vocab Size = %d and Maximum length = %d" % (vocab_size, max_length))

def encoding_doc(token, words):
  return(token.texts_to_sequences(words))

encoded_doc = encoding_doc(word_tokenizer, cleaned_words)

def padding_doc(encoded_doc, max_length):
  return(pad_sequences(encoded_doc, maxlen = max_length, padding = "post"))

padded_doc = padding_doc(encoded_doc, max_length)

padded_doc[:5]

print("Shape of padded docs = ",padded_doc.shape)

#tokenizer with filter changed
output_tokenizer = create_tokenizer(unique_intent, filters = '#"%&()*+,-/:;<=>@[\]^`{|}~')

output_tokenizer.word_index

encoded_output = encoding_doc(output_tokenizer, intent)

encoded_output = np.array(encoded_output).reshape(len(encoded_output), 1)

encoded_output.shape

def one_hot(encode):
  o = OneHotEncoder(sparse = False)
  return(o.fit_transform(encode))

output_one_hot = one_hot(encoded_output)

output_one_hot.shape

train_X, val_X, train_Y, val_Y = train_test_split(padded_doc, output_one_hot, shuffle = True, test_size = 0.3)

print("Shape of train_X = %s and train_Y = %s" % (train_X.shape, train_Y.shape))
print("Shape of val_X = %s and val_Y = %s" % (val_X.shape, val_Y.shape))

# Creating the Model
def create_model(vocab_size, max_length):
  model = Sequential()
  model.add(Embedding(vocab_size, 128, input_length = max_length, trainable = False))
  model.add(Bidirectional(LSTM(128)))
#   model.add(LSTM(128))
  model.add(Dense(32, activation = "relu"))
  model.add(Dropout(0.5))
  model.add(Dense(len(unique_intent), activation = "softmax")) # number of intents

  return model

model = create_model(vocab_size, max_length)

model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
model.summary()

filename = 'model.h5'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

hist = model.fit(train_X, train_Y, epochs = 100, batch_size = 16, validation_data = (val_X, val_Y), callbacks = [checkpoint])

model = load_model("pickles/model.h5")

# summarize history for accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('graphs/accuracy.png', bbox_inches='tight')

# summarize history for loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('graphs/loss.png', bbox_inches='tight')


# Download Functions, Variables, and Model for Inference
pickle.dump(unique_intent, open("pickles/unique_intent.pickle", "wb"))
pickle.dump(word_tokenizer, open("pickles/word_tokenizer.pickle","wb"))
pickle.dump(max_length, open("pickles/max_length.pickle", "wb"))
model.save("pickles/model.h5")
