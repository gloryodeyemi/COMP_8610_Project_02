import pandas as pd
import numpy as np
# from keras.layers import *
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import InputLayer
# from tensorflow.keras.layers import Embedding
# from keras.preprocessing.text import Tokenizer
# from utils.preprocess import *
# from keras.preprocessing.text import Tokenizer
# import keras.layers

import operator
from collections import defaultdict
import warnings
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import pad_sequences
from keras import backend as K
from keras.layers import *
import pickle
import torch
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

class GRU:  
    max_words = 2000
    max_len = 200
    filename = 'gru.pkl'
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Device: ",device)

    def tokenize(self, X, X_train, X_test, X_val):
        tokenizer = Tokenizer(num_words=self.max_words)
        tokenizer.fit_on_texts(list(X))

        train_sequence = tokenizer.texts_to_sequences(X_train)
        train_pad = pad_sequences(train_sequence, maxlen=self.max_len)

        val_sequence = tokenizer.texts_to_sequences(X_val)
        val_pad = pad_sequences(val_sequence, maxlen=self.max_len)

        test_sequence = tokenizer.texts_to_sequences(X_test)
        test_pad = pad_sequences(test_sequence, maxlen=self.max_len)
        
        X_sequence = tokenizer.texts_to_sequences(X)
        X_pad = pad_sequences(X_sequence,maxlen=self.max_len)

        return train_pad, val_pad, test_pad

    def build_model(self):
        print("Building the GRU model...")
        print("-------------------------")
        
        model = models.Sequential()
        model.add(keras.layers.Embedding(self.max_words, 50, input_length=self.max_len, input_shape=[self.max_len]))
        model.add(keras.layers.GRU(64, dropout=0.2))
        model.add(keras.layers.Dense(256, activation='relu'))
        model.add(keras.layers.Dense(6, activation='sigmoid'))
        model.summary()
        
        
        print("Done!")
        
        return model

    def compile_model(self, model):
        print("Compiling the GRU model...")
        print("--------------------------")
        model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.AUC()])

        print("Model compiled!")

        return model

    def train_model(self, model, train_pad, y_train, val_pad, y_val):
        print("Training the model")
        print("------------------")

        history = model.fit(train_pad,
                        y_train,
                        epochs=10,
                        batch_size=512,
                        validation_data=(val_pad, y_val))
        return history

    def plot_accuracy(self, history):
        plt.plot(history.history['accuracy'], label='train')
        plt.plot(history.history['val_accuracy'], label='test')
        plt.legend()
        plt.savefig(f'gru_accuracy.png')
        plt.show()
        
    def plot_precision(self, history):
        plt.plot(history.history['precision'], label='train')
        plt.plot(history.history['val_precision'], label='test')
        plt.legend()
        plt.savefig(f'gru_precision.png')
        plt.show()
        
    def plot_recall(self, history):
        plt.plot(history.history['recall'], label='train')
        plt.plot(history.history['val_recall'], label='test')
        plt.legend()
        plt.savefig(f'gru_recall.png')
        plt.show()
        
    def plot_auc(self, history):
        plt.plot(history.history['auc'], label='train')
        plt.plot(history.history['val_auc'], label='test')
        plt.legend()
        plt.savefig(f'gru_auc.png')
        plt.show()

    # accuracy results
    def get_accuracy(self, model, X_val, y_val, X_train, y_train):    
        print("Results:")
        print("--------")
        train_scores = model.evaluate(X_train, y_train, verbose=False)
        print("Training accuracy: %.2f%%\n" % (train_scores[1] * 100))
        print("Training loss: %.2f%%\n" % (train_scores[0] * 100))
        print("Training precision: %.2f%%\n" % (train_scores[2] * 100))
        print("Training recall: %.2f%%\n" % (train_scores[3] * 100))
#         print("Training f1_score: %.2f%%\n" % (train_scores[4] * 100))
        print("Training auc: %.2f%%\n" % (train_scores[4] * 100))
    
        print("-"*15)
        
        test_scores = model.evaluate(X_val, y_val, verbose=False)
        print("Validation accuracy: %.2f%%\n" % (test_scores[1] * 100))
        print("Validation loss: %.2f%%\n" % (test_scores[0] * 100))
        print("Validation precision: %.2f%%\n" % (test_scores[2] * 100))
        print("Validation recall: %.2f%%\n" % (test_scores[3] * 100))
#         print("Validation f1_score: %.2f%%\n" % (test_scores[4] * 100))
        print("Validation auc: %.2f%%\n" % (test_scores[4] * 100))

    def save_model(self, model):
        # save the model to disk
        pickle.dump(model, open(self.filename, 'wb'))
        print("Model saved!")

    def load_model(self):
        # load the model from disk
        loaded_model = pickle.load(open(self.filename, 'rb'))
        print("Model loaded!")
        return loaded_model

    def make_predictions(self, model, test_pad, col_id, com_text, labels):
        y_pred = model.predict(test_pad)
        pred_result = pd.DataFrame({'id':col_id,
                      'comment_text':com_text,
                      'toxic':y_pred[:,0],
                      'severe_toxic':y_pred[:,1],
                      'obscene':y_pred[:,2],
                      'threat':y_pred[:,3],
                      'insult':y_pred[:,4],
                      'identity_hate':y_pred[:,5]})
        pred_result.to_csv('gru_prediction.csv', index=False)

        return pred_result

