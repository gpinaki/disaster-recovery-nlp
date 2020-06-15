import sys

# python train_classifier.py ../data/DisasterResponse.db classifier.pkl

import pandas as pd
from sqlalchemy import create_engine
import string
import re
import nltk
import string
import numpy as np
from sklearn.model_selection import GridSearchCV
from sqlalchemy import create_engine
nltk.download(['punkt', 'wordnet', 'stopwords'])
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

def load_data(database_filepath):
    
    # Build connection string based on database file path
    connection_string = 'sqlite:///' + database_filepath
    engine = create_engine(connection_string)
    
    # Load a dataframe from the messages table 
    df = pd.read_sql_table('messages',connection_string)
    
    # Seperate feature (X)  and classifications (Y)
    X = df['message']
    Y = df.drop(['message', 'genre', 'id', 'original'], axis=1)
    
    # Get category names
    cat_names = Y.columns.tolist()
    
    return X,Y,cat_names

def tokenize(text):
    
    stop_words = nltk.corpus.stopwords.words("english")
    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
    
    # Remove punctuations, change to lower case
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    
    # tokenize text
    tokens = nltk.word_tokenize(text)
    
    # lemmatize and remove stop words
    return [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]


def build_model():
    
    # Create a RandomForestClassifier object 
    
    clf = RandomForestClassifier()

    # Create a pipeline object
    pipeline = Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultiOutputClassifier(clf))
                    ])

    # create parameter grid
    #'clf__estimator__min_samples_split': [2, 3, 4],
    #'clf__estimator__n_estimators': [50, 100],
    param_grid = {
        'clf__estimator__n_estimators': [50],
        'clf__estimator__min_samples_split': [2]  
      #,  'clf__estimator__criterion': ['entropy', 'gini']
     }

    # Create GridSearchCV object
    
    cv = GridSearchCV(pipeline, param_grid=param_grid,n_jobs=-1,verbose=10)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
  
    Y_pred = model.predict(X_test)
    
    # Calculate model accuracy 
    for i in range(len(category_names)):
        print("Category:", category_names[i],"\n", classification_report(Y_test.iloc[:, i].values, Y_pred[:, i]))
        print('Accuracy of %25s: %.2f' %(category_names[i], accuracy_score(Y_test.iloc[:, i].values, Y_pred[:,i])))

def save_model(model, model_filepath):
    
    joblib.dump(model, model_filepath)


def main():
    
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
