import torch
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
torch.set_default_dtype(torch.float64)
from ast import literal_eval

# VECTORIZER CLASS
class Vectorizer(TfidfVectorizer):
    def __init__(self, ngram_range=None, max_features=None, stop_words=None):

        # vectorizor params
        self.max_features = max_features #vocab size
        self.stopwords = stop_words #if we want to remove these or not
        self.ngram_range = ngram_range #size of ngrams to use as each observation
        # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
        self.tfidf = TfidfVectorizer(
            analyzer='word',
            stop_words=self.stopwords,
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            lowercase=True,
            binary=True
        )
        self.fixed_vocabulary_ = False

    def vectorize_texts(self, texts):
        self.tfidf.fit(texts)
        self.vocabulary_ = self.tfidf.vocabulary_
        return self.tfidf.transform(texts)
    
#REVIEWS DATASET CLASS
class ReviewsDataset(torch.utils.data.Dataset):
    def __init__(self, vectorizer, df_filepath, clean_regex="[^a-zA-Z0-9]", y_col="y", features=None, expanded=False):

        # read in data and encode outcome variable as labels
        self.df = pd.read_csv(df_filepath)

        if expanded == True:
            self.df['reviews'] = self.df['reviews'].apply(literal_eval)
            self.df = self.df.explode("reviews").reset_index()

        self.df[[y_col]] = 0
        self.df.loc[self.df.loc[:,'Overall Compliance'] == "No",y_col] = 1
        self.labels = self.df.loc[:,'y']

        # args for creating a cleaned text
        self.clean_regex = clean_regex
        self.raw_text=self.df['reviews']
        self.cleaned_text = self.clean_texts(self.raw_text)

        # args for vectorizing clean text
        self.vectorizer = vectorizer
        self.text = vectorizer.vectorize_texts(self.cleaned_text).toarray().todense()
        
        if features is not None:
            self.features = self.df.loc[:,features]

    def clean_texts(self, raw_text):
        cleaned = []
        for text in raw_text:
            text = ' '.join(text.split(r'\n'))
            text = re.sub(self.clean_regex," ",text).lower()
            #delete numbers
            text = re.sub(r'\d+', '', text)
            text = re.sub(' +', ' ', text)
            cleaned.append(text)
        return cleaned
    
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