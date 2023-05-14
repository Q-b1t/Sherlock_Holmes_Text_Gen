import torch
import torch.nn as nn
import numpy as np

class PredictionNetwork(nn.Module):
  """
  Parameters 
  vocab_size : int
    THe number of characters in a vocabulary
  
  embedding_dimention :  int
    Dimention of the embedding vectors for each token in the dictionary

  linear_units : int
    Number of hidden units in the linear layer

  lstm_units : int
    Number of hidden units in the lstm layer

  max_norm : int
    If any of the embedding vectors has a higher L2 norm than 'max_norm' it is rescaled.

  n_layers : int
    Number of lstm layers
  """

  def __init__(self,vocab_size,embedding_dimention = 32, linear_units = 64,lstm_units = 8, max_norm = 2,n_layers = 2):
    super().__init__()

    # embedding matrix
    self.embedding_matrix = nn.Embedding(vocab_size,embedding_dimention,padding_idx=vocab_size -1,norm_type=2,max_norm=max_norm)

    # lstm block
    self.lstm_block = nn.LSTM(embedding_dimention,lstm_units,batch_first = True,num_layers = n_layers)

    # classifier
    self.classifier = nn.Sequential(
        nn.Linear(in_features=lstm_units,out_features=linear_units),
        nn.Linear(in_features=linear_units,out_features=vocab_size)
    )

  def forward(self,x,h = None,c = None):
    """
    inputs
    x : torch.Tensor
      Input tensor of shape (batch_size,window_size)
    h,c : torch.Tensor or None
      Hidden states of the LSTM
    
    returns
    logits : torch.Tensor
      Tensor of shape (batch_size,vocab_size)
    
    h,c : torch.Tensor or None
      Hidden states of the LSTM
    """

    emb = self.embedding_matrix(x)
    if h is not None and c is not None:
      _,(h,c) = self.lstm_block(emb,(h,c))
    else:
      _,(h,c) = self.lstm_block(emb)

    h_mean = h.mean(dim = 0)
    logits = self.classifier(h_mean)

    return logits,h,c


def generate_text(n_chars,model,dataset,device,initial_text = "W.A.T.S.O.N", random_state = None):
  res = initial_text
  model.eval()
  h,c = None,None

  if random_state is not None:
    np.random.seed(random_state)
  
  with torch.inference_mode():
    for _ in range(n_chars):
      prev_chars = initial_text if res == initial_text else res[-1]
      features = torch.LongTensor([[dataset.ch2idx[c] for c in prev_chars]]).to(device)
      logits, h,c = model(features,h,c)
      probs = F.softmax(logits[0],dim = 0).to("cpu").detach().numpy()
      new_ch = np.random.choice(dataset.vocabulary,p = probs)
      res += new_ch
  return res
