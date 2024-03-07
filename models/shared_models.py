import torch
from transformers import DistilBertForSequenceClassification


# BERT-based model (only text)
class BERTAndErnie(torch.nn.Module):
    def __init__(self):
        super(BERTAndErnie, self).__init__()
        self.l1 = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2
        )
        self.l2 = torch.nn.Sigmoid()

    def forward(self, ids, mask):
        out = self.l1(ids, attention_mask=mask)
        output = self.l2(out.logits)

        return output


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
