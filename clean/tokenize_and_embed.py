import os, re, sys
import pandas as pd
# from xgboost import XGBClassifier

# #sci-kit learn
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve, roc_auc_score

# #nltk
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import word_tokenize, wordpunct_tokenize
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')


def read_data(filepath):
    data = pd.read_csv(filepath)

    data[['y']] = 0
    data.loc[data.loc[:,'Overall Compliance'] == "No",'y'] = 1

    #feature selection (come back to this if needed)
    data_simple = data[["reviews", "ratings", "n_reviews", "avg_rating", "y"]]

    return data_simple


def tokenize_and_embed(reviews_series, type):
    '''
    Inputs: Series of reviews
    Outputs: Df of floats - matrix with embedded text
    '''

    if type == "td-idf":
        ## Encode text data with n_gram = 2
        vectorizer2 = TfidfVectorizer(analyzer='word', 
                                    ngram_range=(1, 2),
                                    stop_words='english',
                                    binary=True)

        X2_train = vectorizer2.fit_transform(reviews_series)
        X2_train = X2_train.todense()
        X2_train_pd = pd.DataFrame(X2_train)

        return X2_train_pd
    
    if type == "glove":
        pass


def main(filepath="data/phila/labeled_inspections_with_reviews.csv", type=None):
    
    type="td-idf"

    df = read_data(filepath)
    tokenized_reviews = tokenize_and_embed(df['reviews'], type)
    print(tokenized_reviews.head())


if __name__ == "__main__":
    #filepath = sys.argv[1]
    main()

