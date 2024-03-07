import torch
from transformers import DistilBertForSequenceClassification


# BERT-based model (only text)
class TextBERT(torch.nn.Module):
    def __init__(self):
        super(TextBERT, self).__init__()
        self.l1 = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2
        )
        self.l2 = torch.nn.Sigmoid()

    def forward(self, ids, mask):
        out = self.l1(ids, attention_mask=mask)
        output = self.l2(out.logits)

        return output


class FullBERT(torch.nn.Module):
    def __init__(self, num_features):
        super(FullBERT, self).__init__()
        self.l1 = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2
        )
        self.l2 = torch.nn.Linear(2 + num_features, 2)
        self.l3 = torch.nn.Sigmoid()

    def forward(self, ids, mask, features):
        out1 = self.l1(ids, attention_mask=mask)
        out = self.l2(torch.cat((out1.logits, features), dim=1))

        return self.l3(out)


## RNN-LSTM
class RNNLM(torch.nn.Module):
    """Container module with an embedding module, an LSTM module,
    and a final linear layer to map the LSTM output to the
    vocabulary.
    """

    def __init__(self, embedding_dim, hidden_dim, num_layers, num_labels, dropout=0.5):
        super(RNNLM, self).__init__()

        self.dropout = torch.nn.Dropout(dropout)
        self.lstm = torch.nn.LSTM(
            embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )
        self.llayer = torch.nn.Linear(hidden_dim, num_labels)
        self.activation = torch.nn.Sigmoid()

    def forward(self, input, hidden0):
        """
        Run forward propagation for a given minibatch of inputs using
        hidden0 as the initial hidden state"
        """
        embeds = self.dropout(input)
        lstm_out, hiddenn = self.lstm(embeds, hidden0)
        l_out = self.dropout(lstm_out)
        output = self.llayer(l_out[:, -1, :])

        return self.activation(output), hiddenn


## Non-text feature list
FEATURES = [
    "stars",
    "review_count",
    "is_open",
    "n_reviews",
    "avg_rating",
    "IR_regular",
    "IR_follow_up",
    "IR_other",
    "Chester",
    "Bucks",
    "Philadelphia",
    "Delaware",
    "Montgomery",
    "Berks",
    "Nightlife",
    "Bars",
    "Pizza",
    "Italian",
    "Sandwiches",
    "Breakfast & Brunch",
    "Cafes",
    "Burgers",
    "Delis",
    "Caterers",
    "Mexican",
    "Desserts",
    "Salad",
    "Sports Bars",
    "Pubs",
    "Chicken Wings",
    "Seafood",
    "Beer",
    "Wine & Spirits",
    "Juice Bars & Smoothies",
    "Mediterranean",
    "Gastropubs",
    "Diners",
    "Steakhouses",
    "Breweries",
    "Donuts",
    "Barbeque",
    "Cheesesteaks",
    "Middle Eastern",
    "Wineries",
    "Indian",
    "Halal",
    "Vegan",
    "Vegetarian",
    "Beer Bar",
    "Soup",
    "Sushi Bars",
]
