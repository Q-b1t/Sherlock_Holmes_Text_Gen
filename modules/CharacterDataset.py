import torch
from torch.utils.data import Dataset
from collections import Counter,defaultdict

class CharacterDataset(Dataset):
  """
  text: str
    Input text that will be used to create the dataset's vocabulary
  window size: int
    Number of characters of a sequence to use as input features to make a prediction
  vocab size: int
    NUmber of characters in the vocabulary. The last one will be a default for characters absent in the vocabulary
  
  Attributes
  ch2idx: 
    Mapping of the character to the index in the vocabulary. The characters that are unknown.
  idx2ch:
    Mapping of the index on the vocabulary to the character
  vocabulary: list
    List of all the vocabulary
  """
  def __init__(self, text, window_size=1, vocab_size=50):
    super().__init__()
    assert len(list(set(text))) > vocab_size, "the vocabulary size must be smaller than the number of tokens in the text"
    # instance the class attributes
    self.text = text.replace("\n", " ")
    self.window_size = window_size
    self.ch2idx = defaultdict(lambda: vocab_size - 1)

    # create the vocabulary based on the token frequency in the text. Truncate the vocabulary to the specified length
    most_common_ch2idx = {
         x[0]: i
         for i, x in enumerate(Counter(self.text).most_common()[: (vocab_size - 1)])
    }
    self.ch2idx.update(most_common_ch2idx)
    
    # add the uknown vocaculary
    self.ch2idx["~"] = vocab_size - 1

    # create the reverse dictionary
    self.idx2ch = {v: k for k, v in self.ch2idx.items()}

    # create the vocabulary
    self.vocabulary = [self.idx2ch[i] for i in range(vocab_size)]

  def __len__(self):
    return len(self.text) - self.window_size

  def __getitem__(self, ix):
    X = torch.LongTensor([self.ch2idx[c] for c in self.text[ix : ix + self.window_size]])
    y = self.ch2idx[self.text[ix + self.window_size]]
    return X, y
