
from tensorflow.keras.models import load_model
import tensorflow_hub as hub
from numpy import asarray,array, argmax, expand_dims, squeeze
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import os

embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
model=load_model(os.path.join(os.path.abspath(os.path.dirname(__file__)),"Question_Classifier.h5"))


class Clause:


    def __init__(self):
        self.distinct_types ={0: 'SELECT DISTINCT {} FROM {}', 1: 'SELECT MAX(DISTINCT {}) FROM {}', 2: 'SELECT MIN(DISTINCT {}) FROM {}',
                      3: 'SELECT COUNT(DISTINCT {}) FROM {}', 4: 'SELECT SUM(DISTINCT {}) FROM {}', 5: 'SELECT AVG(DISTINCT {}) FROM {}'}

        self.types = {0: 'SELECT {} FROM {}', 1: 'SELECT MAX({}) FROM {}', 2: 'SELECT MIN({}) FROM {}',
                      3: 'SELECT COUNT({}) FROM {}', 4: 'SELECT SUM({}) FROM {}', 5: 'SELECT AVG({}) FROM {}'}

    def get_embeddings(self, x):
        embeddings = embed(x)
        return asarray(embeddings)

    def testEmb(self, q, inttype=False, summable=False,distinct=False):
        emb = self.get_embeddings(q)
        #self.clause = argmax(model.predict(emb))
        #return model.predict(emb)
    
        # Make predictions
    
        # Remove the extra dimension from the input data
        #input_data = squeeze(emb, axis=1)

        # Make predictions
       # num_predictions = 5  # Number of predictions you want to obtain
        #predictions = []

        #for _ in range(num_predictions):
        #    prediction = model.predict(expand_dims(input_data, axis=0))
         #   predictions.append(prediction)

        # Convert the list of predictions to a NumPy array
        #predictions = array(predictions)

        predictions = model.predict(emb)
        #expected_labels=[0,1,2,3,4,5]
        # Convert predictions to class labels
        predicted_labels = argmax(predictions, axis=1)

        # Calculate evaluation metrics
        #accuracy = accuracy_score(expected_labels, predicted_labels)
        #precision = precision_score(expected_labels, predicted_labels, average='weighted')
        #recall = recall_score(expected_labels, predicted_labels, average='weighted')
        #f1 = f1_score(expected_labels, predicted_labels, average='weighted')

        # Print the evaluation metrics
        #print("Accuracy:", accuracy)
        #print("Precision:", precision)
        #print("Recall:", recall)
        #print("F1 Score:", f1)
        #return [accuracy,precision,recall,f2]
        return [predictions,predicted_labels]

    def adapt(self, q, inttype=False, summable=False,distinct=False):
        emb = self.get_embeddings(q)
        #self.clause = argmax(model.predict(emb))
        
        
        if distinct:
            self.clause = self.distinct_types[argmax(model.predict(emb))]
        else:
            self.clause = self.types[argmax(model.predict(emb))]

        if summable and inttype and "COUNT" in self.clause:
            self.clause = '''SELECT SUM({}) FROM {}'''
        
        return self.clause
