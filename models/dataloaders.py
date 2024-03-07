import torch
import pandas as pd
import re
from typing import Callable
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import DistilBertTokenizer
from torchtext.vocab import GloVe
from ast import literal_eval

torch.set_default_dtype(torch.float64)
from torch.utils.data import Dataset
from torchtext.data import get_tokenizer


# VECTORIZER CLASS - Claire
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
    
#REVIEWS DATASET CLASS - Claire
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
        self.text = vectorizer.vectorize_texts(self.cleaned_text).toarray()
        
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


# BERT dataset - Jack
class BERTReviewData(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: DistilBertTokenizer,
        max_tokens: int,
        expanded: bool = False,
    ):
        self.tokenizer = tokenizer
        self.df = df
        self.max_tokens = max_tokens
        self.expanded = expanded
        self.review_text = self.clean_text(self.df)
        self.target_cat = self.df["Overall Compliance"]

    def clean_text(self, df: pd.DataFrame) -> pd.Series:

        def clean_reviews(reviews):
            cleaned = []
            for review in reviews:
                review = review.replace("\n", " ")
                cleaned.append(
                    re.sub(r"[^a-zA-Z0-9]", " ", review).strip()
                )  # may need to find a better way to do so

            return cleaned

        if self.expanded:
            df["reviews"] = df["reviews"].str.strip()
            df["reviews"] = df["reviews"].str.replace("\n", " ")
            df["reviews"] = df["reviews"].str.replace(r"[^a-zA-Z0-9]", " ", regex=True)

            return df["reviews"]

        return df["reviews"].apply(clean_reviews)

    def __len__(self):
        return len(self.review_text)

    def __getitem__(self, index):
        review_text = str(self.review_text[index])
        target_cat = self.target_cat[index]

        if not self.expanded:
            # combine all reviews into one string
            review_text = " ".join(self.review_text[index])

        inputs = self.tokenizer.encode_plus(
            review_text,
            None,
            add_special_tokens=True,
            max_length=self.max_tokens,
            padding="max_length",
            truncation=True,
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        # [0, 1] = pass, [1, 0] = fail
        target = []
        if target_cat == "No":
            target = [1, 0]
        else:
            target = [0, 1]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "targets": torch.tensor(target, dtype=torch.float),
        }


# Fake review dataset - Jack
class FakeReviewData(Dataset):
    def __init__(self, df: pd.DataFrame, max_tokens: int, embedding: Callable):
        self.embedding = embedding
        if not embedding:
            self.embedding = GloVe("6B")
        self.tokenizer = get_tokenizer("basic_english")
        self.df = df
        self.max_tokens = max_tokens
        self.review_text = self.clean_text(self.df)
        self.target_cat = self.df["Overall Compliance"]

    def clean_text(self, df: pd.DataFrame) -> pd.Series:

        df["text"] = df["text"].str.strip()
        df["text"] = df["text"].str.replace("\n", " ")
        df["text"] = df["text"].str.replace(r"[^a-zA-Z0-9\s]", "", regex=True)

        return df["text"]

    def __len__(self):
        return len(self.review_text)

    def __getitem__(self, index):
        review_text = str(self.review_text[index])
        tokens = self.tokenizer(review_text)[: self.max_tokens]
        target_cat = self.target_cat[index]
        inputs = self.embedding.get_vecs_by_tokens(tokens)

        # [0, 1] = real, [1, 0] = gpt
        target = []
        if target_cat == "Yes":
            target = [1, 0]
        else:
            target = [0, 1]

        padding = torch.nn.ZeroPad2d((0, 0, 0, self.max_tokens - len(inputs)))

        return {
            "text": padding(inputs),
            "label": torch.tensor(target, dtype=torch.float),
        }


# OTHER HELPERS - Claire
def encode_output_variable(filepath, svm=None):
    df = pd.read_csv(filepath)
    if svm:
        df[["y"]] = -1
    else:
        df[["y"]] = 0
    df.loc[df.loc[:, "Overall Compliance"] == "No", "y"] = 1
    return df
