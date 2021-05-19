import torch
import torch.nn as nn

class CBoWTextClassifier(nn.Module):
  '''
  Simple text classifier by using Continous Bag of Word (CBoW)
  Reference : http://www.cse.chalmers.se/~richajo/nlp2019/l2/Text%20classification%20using%20a%20CBoW%20representation.html
  '''
  def __init__(self, vocab_size, num_labels, embed_dim, hidden_size=20, dropout_ratio=0.1):
    super().__init__()
    self.vocab_size = vocab_size
    self.num_labels = num_labels
    self.embedding = nn.Embedding(vocab_size, embed_dim)
    self.hidden_layer = nn.Linear(embed_dim, hidden_size)
    self.output = nn.Linear(hidden_size, num_labels)
    self.dropout_ratio = nn.Dropout(dropout_ratio)

  def forward(self, docs):
    # First look up the embeddings for all the words in the documents.
    embedded = self.embedding(docs) # The shape is now (max_len, n_docs, emb_dim)
    # Compute the mean of word embeddings over the documents.
    pooling = embedded.mean(dim = 0)     # The shape is now (n_docs, emb_dim)
    # Apply the dropout layer. (This is only used during training, not during testing.)
    dropout = self.dropout(pooling)
    hidden = torch.relu(self.hidden_layer(dropout))
    preds = self.output(hidden)

    return preds