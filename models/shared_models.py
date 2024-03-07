import torch
from transformers import DistilBertForSequenceClassification
from torch import nn
import torch.nn.functional as F

# BERT-based model (only text) - Jack
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


## RNN-LSTM - Jack
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

# Logistic Regression - Claire
class LogisticRegression(torch.nn.Module):
   def __init__(self, input_dim, output_dim):
      super(LogisticRegression, self).__init__()
      self.linear = torch.nn.Linear(input_dim, output_dim)
      self.nonlinearity = torch.nn.Sigmoid()
   
   def forward(self, x):
      z = self.linear(x)
      #z_prime = self.linear_two(z)
      y_hat = self.nonlinearity(z)
      return y_hat
   
# Logistic Regression with Features - Claire
class LogisticRegressionwithFeatures(torch.nn.Module):
   def __init__(self, input_dim, feature_dim, output_dim):
      super(LogisticRegressionwithFeatures, self).__init__()
      self.linear = torch.nn.Linear(input_dim, 2)
      self.sigmoid = torch.nn.Sigmoid()
      self.feature_layer = torch.nn.Linear(feature_dim+2, output_dim)
      
   def forward(self, x, features):
      z = self.linear(x)
      x_prime = self.sigmoid(z)
      z_features = torch.cat((x_prime, features), 1)
      z_prime = self.feature_layer(z_features)
      y_hat = torch.sigmoid(z_prime)
      return y_hat
   
#SVM - Claire
class SVM(nn.Module):
    def __init__(self, n_features):
        super(SVM, self).__init__()
        self.linear = nn.Linear(n_features, 2)
    
    def forward(self, x):
        out = self.linear(x)
        return out