# adapted from https://stackabuse.com/text-classification-with-python-and-scikit-learn/

# IMPORTS
import numpy as np  
import pandas as pd
import re  
import nltk  
from sklearn.datasets import load_files  
nltk.download('stopwords')  
import pickle  
from nltk.corpus import stopwords 
nltk.download('wordnet')

# STEMMING / FILTERING FUNCTION
def filterDocument(X):
    documents = []
    from nltk.stem import WordNetLemmatizer
    stemmer = WordNetLemmatizer()
    for sen in range(0, len(X)):  
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(X.iloc[sen]))
        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)
        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)
        # Converting to Lowercase
        document = document.lower()
        # Lemmatization
        document = document.split()
        document = [stemmer.lemmatize(word) for word in document]
        document = ' '.join(document)
        documents.append(document)
    return documents


# RETRIEVE CSV DATASET
full_cols = ['target','ids','date','flag','user','text']
my_df = df = pd.read_csv("./training.csv",header=None,encoding='latin-1',names=full_cols)

# RETRIEVING ENTIRE DATASET CAN CAUSE MEMORY ERROR
limit = my_df.sample(1000)
X,y = limit.text, limit.target
documents = filterDocument(X)

#**Modèle choisi**: 'Bag of Words'
#**max_features**: nombre maximum de mots utilisés (on filtre les mots peu fréquents)
#**min_df**: au moins n documents doivent contenir le mot.
#**max_df**: pas plus de n% de documents doivent contenir le mot.
#**stop_words**: on enlève les stop words.
#**fit_transform**: conversion text => données numériques
from sklearn.feature_extraction.text import CountVectorizer  
vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))  
X = vectorizer.fit_transform(documents).toarray()  


# DEVIDE DATA INTO TRAIN AND TEST
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  


import os.path
if(os.path.exists('text_classifier')):
    #GET MODEL FROM DISK
    with open('text_classifier', 'rb') as training_model:  
        classifier = pickle.load(training_model)
else:
    #MODEL WASN'T FOUND DO CLASSIFICATION
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators=1000, random_state=0)  
    classifier.fit(X_train, y_train) 

# MAKE A PREDICTION
y_pred = classifier.predict(X_test)  

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# SHOW RESULTS
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))  
print(accuracy_score(y_test, y_pred)) 


# SAVE MODEL TO DISK
with open('text_classifier', 'wb') as picklefile:  
    pickle.dump(classifier,picklefile)