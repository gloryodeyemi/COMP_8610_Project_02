from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import pad_sequences

import pickle
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class LSTM:  
    max_words = 2000
    max_len = 200
    filename = 'lstm.pkl'
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
        print("Building the model...")
        print("---------------------")
        
        model = models.Sequential()
        model.add(layers.Embedding(self.max_words, 50, input_length=self.max_len, input_shape=[self.max_len]))
        model.add(layers.LSTM(64, dropout=0.2))
        # model.add(layers.dropout=0.2)
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(6, activation='sigmoid'))
        model.summary()
        
        print("Done!")
        
        return model

    def compile_model(self, model):
        print("Compiling the model...")
        print("----------------------")
        model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])

        print("Model compiled!")

        return model

    def train_model(self, model, train_pad, y_train, val_pad, y_val):
        print("Training the model")
        print("------------------")

        history = model.fit(train_pad,
                        y_train,
                        epochs=5,
                        batch_size=512,
                        validation_data=(val_pad, y_val))
        return history

    def plot_accuracy(self, history):
        plt.plot(history.history['accuracy'], label='train')
        plt.plot(history.history['val_accuracy'], label='test')
        plt.legend()
        plt.savefig(f'accuracy.png')
        plt.show()

    # accuracy results
    def get_accuracy(self, model, X_val, y_val, X_train, y_train):    
        print("Results:")
        print("--------")
        train_scores = model.evaluate(X_train, y_train, verbose=False)
        print("Training Accuracy: %.2f%%\n" % (train_scores[1] * 100))
        print("Training loss: %.2f%%\n" % (train_scores[0] * 100))
        test_scores = model.evaluate(X_val, y_val, verbose=False)
        print("Validation accuracy: %.2f%%\n" % (test_scores[1] * 100))
        print("Validation loss: %.2f%%\n" % (test_scores[0] * 100))

    def save_model(self, model):
        # save the model to disk
        pickle.dump(model, open(self.filename, 'wb'))
        print("Model saved!")

    def load_model(self):
        # load the model from disk
        loaded_model = pickle.load(open(self.filename, 'rb'))
        print("Model loaded!")
        return loaded_model

    def make_predictions(self, model, test_pad, col_id, labels):
        y_pred = model.predict(test_pad)
        pred_result = pd.DataFrame({'id':col_id,
                      'toxic':y_pred[:,0],
                      'severe_toxic':y_pred[:,1],
                      'obscene':y_pred[:,2],
                      'threat':y_pred[:,3],
                      'insult':y_pred[:,4],
                      'identity_hate':y_pred[:,5]})
        pred_result.to_csv('prediction.csv', index=False)

        return pred_result