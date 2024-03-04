import torch
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
torch.set_default_dtype(torch.float64)

# VECTORIZER CLASS
class Vectorizer(object):
    def __init__(self, df_filepath, ngram_range=None, clean_regex=None, max_features=None, stop_words=None):
        # read in reviews to vectorizor object
        self.df = pd.read_csv(df_filepath)
        self.texts=self.df['reviews']

        # vectorizor params
        self.clean_regex = clean_regex
        self.max_features = max_features #vocab size
        self.stopwords = stop_words #if we want to remove these or not
        self.ngram_range = ngram_range #size of ngrams to use as each observation
        self.tfidf = TfidfVectorizer(analyzer='word',
                                     stop_words=self.stopwords,
                                     ngram_range=self.ngram_range,
                                     max_features=self.max_features)

    def clean_texts(self):
        cleaned = []
        for text in self.texts:
            if self.clean_regex is not None:
                text = re.sub(self.clean_regex," ",text)
            text = text.lower().strip()
            cleaned.append(text)
        return cleaned

    def set_tfidf(self,cleaned_texts):
        self.tfidf.fit(cleaned_texts)

    def build_vectorizer(self):
        cleaned_texts = self.clean_texts()
        self.set_tfidf(cleaned_texts)
        
    def vectorizeTexts(self):
        cleaned_texts = self.clean_texts()
        return self.tfidf.transform(cleaned_texts)

#REVIEWS DATASET CLASS
class ReviewsDataset(torch.utils.data.Dataset):
    def __init__(self, vectorizer, df_filepath, max_features=7000,
                 ngram_range=(1,2), y_col="y", features=None):

        # read in data and encode outcome variable
        self.df = pd.read_csv(df_filepath)
        self.df[[y_col]] = 0
        self.df.loc[self.df.loc[:,'Overall Compliance'] == "No",y_col] = 1
        
        #self.text = vectorized_reviews
        self.labels = self.df.loc[:,'y']
        
        if features is not None:
            self.features = self.df.loc[:,features]

        #self.text = vectorized_reviews
        self.vectorizer = vectorizer(df_filepath=df_filepath,
                        clean_regex="[^a-zA-Z0-9]",
                        max_features=max_features,
                        ngram_range=ngram_range,
                        stop_words="english")
        self.vectorizer.build_vectorizer()
        self.text = self.vectorizer.vectorizeTexts().toarray()
        self.feature_labels = list(self.features.columns)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        sample = {}
        sample["text"] = self.text[idx]
        sample["labels"] = self.labels[idx]
        if self.features is not None:
            sample["features"] = self.features.iloc[idx,:].values
        return sample
    

#OTHER HELPERS
def encode_output_variable(filepath, svm=None):
    df = pd.read_csv(filepath)
    if svm:
        df[['y']] = -1
    else:
        df[['y']] = 0
    df.loc[df.loc[:,'Overall Compliance'] == "No",'y'] = 1
    return df