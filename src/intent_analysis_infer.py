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
import csv

import time
from tqdm import tqdm

nltk.download("stopwords")
nltk.download("punkt")

# load Trained Functions and Model
unique_intent = pickle.load(open("pickles/unique_intent.pickle", "rb"))
word_tokenizer = pickle.load(open("pickles/word_tokenizer.pickle","rb"))
max_length = pickle.load(open("pickles/max_length.pickle", "rb"))
model = load_model("pickles/model.h5")

def load_dataset(filename):
  df = pd.read_csv(filename, encoding = "latin1", names = ["Comment"])
  df.dropna(subset = ["Comment"], inplace=True)
  comments = list(df["Comment"])

  return (comments)

comments = load_dataset("data/infer.csv")
inference = list()
probability = list()

def padding_doc(encoded_doc, max_length):
  return(pad_sequences(encoded_doc, maxlen = max_length, padding = "post"))

# Predictive Functions
def predictions(text):
  clean = re.sub(r'[^ a-z A-Z 0-9]', " ", text)
  test_word = word_tokenize(clean)
  test_word = [w.lower() for w in test_word]
  test_ls = word_tokenizer.texts_to_sequences(test_word)

  # Remove following # to print tokenized comment
  #print(test_word)

  #Check for unknown words
  if [] in test_ls:
    test_ls = list(filter(None, test_ls))

  test_ls = np.array(test_ls).reshape(1, len(test_ls))

  x = padding_doc(test_ls, max_length)

  pred = model.predict(x)


  return pred

def get_final_output(pred, classes):
  predictions = pred[0]

  classes = np.array(classes)
  ids = np.argsort(-predictions)
  classes = classes[ids]
  predictions = -np.sort(-predictions)

  # Remove the # for next 3 lines to print Intent Probability Distribution for each comment
  #for i in range(pred.shape[1]):
    #print("%s has confidence = %s" % (classes[i], (predictions[i])))

  #print("FINAL: %s has confidence = %s" % (classes[0], (predictions[0])))

  return (classes[0], predictions[0])

# Making Predictions
for comm in comments:
    pred = predictions(comm)
    intent_pred, predict = get_final_output(pred, unique_intent)
    inference.append(intent_pred)
    probability.append(predict)

output_file = open('data/inference_output.csv', 'w', newline = '')

with output_file:
    # identifying header
    headers = ['Comment', 'Intent', 'Confidence']
    writer = csv.DictWriter(output_file, fieldnames = headers)

    # writing data row-wise into the csv file
    writer.writeheader()
    i = 0
    for com in tqdm(comments):
        time.sleep(2)
        writer.writerow({'Comment' : comments[i],
                         'Intent': inference[i],
                         'Confidence': probability[i]})
        i += 1
