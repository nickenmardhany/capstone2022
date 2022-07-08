import re
import string
from unittest import result
import nltk
import pickle
import pandas as pd
import numpy as np
import dill
import os
import json
import torch
import transformers
import pandas as pd
from typing import Any, Dict, List, Callable, Optional, Tuple, Union
from transformers import BertModel, BertTokenizer, DistilBertModel, DistilBertTokenizer
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from torch import optim, nn
from sklearn.base import BaseEstimator, TransformerMixin
import seaborn as sns
sns.set()


#load model
path = 'pipeline_bert_svm.pkl'
clf1 = pickle.load(open(path, 'rb'))
data1 = pd.read_csv("test.csv", encoding='latin1')
data1.columns = ["Label", "Tweet"]


#Preprocessing 

def clean1(isi):
##URl
  isi = re.sub(r"htt\S{0,}", " ",isi)
##USER
  isi = re.sub(r'@\S{0,}', ' ',isi)
  isi = re.sub(r"\s+", " ", isi)
##LOWER
  isi = isi.lower()
##ANGKA
  isi = re.sub(r"\d+", "", isi)
#TANDA BACA
  isi = isi.translate(str.maketrans("","",string.punctuation))
##stop(isi):
  isi = re.sub('[^a-zA-Z0-9\n\.]', ' ', isi)

  return isi

class BertTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        tokenizer,
        model,
        max_length: int = 60,
        embedding_func: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.model.eval()
        self.max_length = max_length
        self.embedding_func = embedding_func

        if self.embedding_func is None:
            self.embedding_func = lambda x: x[0][:, 0, :].squeeze()

        # TODO:: PADDING

    def _tokenize(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        # Tokenize the text with the provided tokenizer
        tokenized_text = self.tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=self.max_length, truncation=True
        )["input_ids"]

        # padding
        padded_text = tokenized_text + [0]*(self.max_length-len(tokenized_text))
        # Create an attention mask telling BERT to use all words
        attention_mask = np.where(np.array(padded_text) != 0, 1, 0)

        # bert takes in a batch so we need to unsqueeze the rows
        return (
            torch.tensor(padded_text).unsqueeze(0),
            torch.tensor(attention_mask).unsqueeze(0),
        )

    def _tokenize_and_predict(self, text: str) -> torch.Tensor:
        tokenized, attention_mask = self._tokenize(text)

        embeddings = self.model(tokenized, attention_mask)
        return self.embedding_func(embeddings)

    def transform(self, text: List[str]):
        if isinstance(text, pd.Series):
            text = text.tolist()

        with torch.no_grad():
            return torch.stack([self._tokenize_and_predict(string) for string in text])
        
Vector = BertTransformer(BertTokenizer.from_pretrained("indobenchmark/indobert-base-p1"),
                      BertModel.from_pretrained("indobenchmark/indobert-base-p2"),
                      embedding_func=lambda x: x[0][:, 0, :].squeeze())


def predict():
    
    data1['baru'] = data1['Tweet'].apply(lambda x: clean1(x))
    vectorized_data = Vector.transform(data1.baru)
    result = clf1.predict(vectorized_data)
    
    print(result)
    return result


predict()