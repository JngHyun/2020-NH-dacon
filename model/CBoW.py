import torch
import torch.nn as nn


class CBoWTextClassifier(nn.Module):
    """
    Simple text classifier by using Continous Bag of Word (CBoW)
    Reference :
    http://www.cse.chalmers.se/~richajo/nlp2019/l2/Text%20classification%20using%20a%20CBoW%20representation.html
    """

    def __init__(
        self, vocab_size, num_label, emb_dim, hidden_size=20, dropout_ratio=0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_label = num_label
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.hidden_layer = nn.Linear(emb_dim, hidden_size)
        self.output = nn.Linear(hidden_size, num_label)
        self.dropout= nn.Dropout(dropout_ratio)

    def forward(self, docs):
        embedded = self.embedding(docs)
        pool_out = embedded.mean(dim=0)
        dropout = self.dropout(pool_out)
        hidden = torch.relu(self.hidden_layer(dropout))
        preds = self.output(hidden)

        return preds
