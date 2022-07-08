from typing import Any, Dict, List, Callable, Optional, Tuple, Union
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from auth import AuthHandler
from schemas import AuthDetails, Login
from database import SessionLocal
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from pydantic import BaseModel
import re
import string
import nltk
import pickle
import pandas as pd
import models
import requests
import models
import json
import re
import string
from unittest import result
import numpy as np
import dill
import os
import torch
import transformers
from transformers import BertModel, BertTokenizer, DistilBertModel, DistilBertTokenizer
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from torch import optim, nn
from sklearn.base import BaseEstimator, TransformerMixin
import seaborn as sns
sns.set()

app = FastAPI()

# #: Configure CORS
# origins = [
#     "http://localhost:5001",
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


#: Initialize model

class Tweets(BaseModel):
    id: int
    tweets: str
    class Config:
            orm_mode=True

class Data(BaseModel):
    id: int
    tweets: str
    user: str
    mark : str
    label: str
    category: str

    class Config:
            orm_mode=True

class User(BaseModel):
    id: str
    first_name: str
    last_name: str
    email: str
    username: str
    role: str
    class Config:
        orm_mode=True

class InsertData(BaseModel):
    tweets: str
    user: str
    mark : str
    label: str
    category: str

db=SessionLocal()

Data = List[Data]


#: Describe all Pydantic Response classes
class ResponseBase(BaseModel):
    status: str
    code: int
    messages: List[str] = []

class DummyResponse(ResponseBase):
    data: Data

class TweetsResponse(ResponseBase):
    data: Tweets

class ListTweetsResponse(ResponseBase):
    data: List[Tweets]

class ListDummyResponse(ResponseBase):
    data: List[Data]

#Load Model
nltk.download('punkt')

path = 'pipeline.pkl'
clf1 = pickle.load(open(path, 'rb'))


#: Test CRUD
@app.get("/api")
def index():
    return {
        "status": "ok",
        "code": 200,
        "data": "Welcome, please check /docs or /redoc",
    }

#Authentication

auth_handler = AuthHandler()
users = []

@app.post('/register', status_code=201)
def register(auth_details: AuthDetails):
    if any(x['username'] == auth_details.username for x in users):
        raise HTTPException(status_code=400, detail='Username is taken')
    
    if any(x['email'] == auth_details.email for x in users):
        raise HTTPException(status_code=400, detail='email is already registered')
    
    hashed_password = auth_handler.get_password_hash(auth_details.password)
    users.append({
        'username': auth_details.username,
        'email' : auth_details.email,
        'first_name':auth_details.first_name,
        'last_name' : auth_details.last_name,
        'password': hashed_password 
        
    })

    
    return {
        "status": "success",
        "code": 200,
        "messages": ["User has been registered !"],
        "data": users,
    }


@app.post('/login')
def login(auth_details: Login):
    user = None
    for x in users:
        if x['email'] == auth_details.email:
            user = x
            break
    
    if (user is None) or (not auth_handler.verify_password(auth_details.password, user['password'])):
        raise HTTPException(status_code=401, detail='Invalid username and/or password')
    token = auth_handler.encode_token(user['email'])
    return { 
        "status": "success",
        "code": 200,
        "messages": ["Login successfully!"],
        "token": token,
        "data": user,
         }


@app.get('/unprotected')
def unprotected():
    return { 'hello': 'world' }


@app.get('/users/me')
def protected(username=Depends(auth_handler.auth_wrapper)):
    return { 'name': username }

#: Twitter Streaming 

def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """

    r.headers["Authorization"] = f"Bearer AAAAAAAAAAAAAAAAAAAAALRQVwEAAAAAMnmA%2BU3ltzAl6vyoYBNLH06SUUQ%3DiCkOdnS92bQnu8qAEADDMR8f0pBJhXcorJXgxBzYeby9BLRuEw"
    r.headers["User-Agent"] = "v2FilteredStreamPython"
    return r


@app.get("/rules")
async def get_rules_stream():
    response = requests.get(
        "https://api.twitter.com/2/tweets/search/stream/rules", auth=bearer_oauth
    )
    return response.json()

@app.get("/stream")
async def get_stream():
    response = requests.get(
        "https://api.twitter.com/2/tweets/search/stream?expansions=author_id", auth=bearer_oauth, stream=True,
    )
    for response_line in response.iter_lines():
        if response_line:
            json_response = json.loads(response_line)
            if  json_response != '':
                tweeet = json_response['data']['text']
                tweets = tweeet.encode("utf-8")
                new_tweet=models.Data(
                
                users = json_response['includes']['users'][0]['username'],
                tweets=json_response['data']['text'],
                mark='unprocessed',
                label='unlabelled',
                category='uncategorized'
                )
        
                
                db.add(new_tweet)
                db.commit()
            # print(json.dumps(tweets.decode("utf-8"), indent=4, sort_keys=True))
            print(json.dumps(json_response, indent=4, sort_keys=True))

    return {"status": "ok", "code": 200, "data": tweets.decode("utf-8")} 

#MODEL


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




#method ade if output = 1, add to database pengaduan

def process(data: Data):

      clean_data = [clean1(data.tweets)]
    #   vectorized_data = Vector.transform(clean_data)
      result = clf1.predict(clean_data)
      if result == 1:
        data.label = "pengaduan"
      else:
        data.label = "bukan pengaduan"

      db.add(data)
      db.commit()
      
      data2=db.query(models.Data).all()
      print(data2)
      return { 
          "status": "success",
          "code": 200,
          "messages": ["Data processed!"],
          "data": data2
          }

@app.get("/data", response_model=ListDummyResponse)
async def get_all_data():
   tweet=db.query(models.Data).all()
   for x in tweet:
            if x.mark == "unprocessed":
                #panggil method ade (inputan data berupa single data, terdiri dari semua column)
                process(x)

                #labeling as processed
                x.mark = "processed"
                db.add(x)
                db.commit()
                
                print(x)
  
   return {"status": "ok", "code": 200, "data": tweet}


@app.get("/data/pengaduan", response_model=ListDummyResponse)
async def get_data_pengaduan():
   data=db.query(models.Data).filter(models.Data.label=='pengaduan').all()
   list = sorted(data)
   return {"status": "ok", "code": 200, "data": list}


@app.post("/data", status_code=201, response_model=DummyResponse)
def create_data_pengaduan(user: InsertData):
    new_user=models.Data( tweets=user.tweets,users=user.users,mark=user.mark,label= user.label,category = user.category)

    db.add(new_user)
    db.commit()
    return {
        "status": "success",
        "code": 201,
        "messages": ["Book added !"],
        "data": new_user,
    }

@app.get("/predict_data")
async def predict_laporan():
    data=db.query(models.Data).all()
      
    for x in data:
            if x.mark == "unprocessed":
                #panggil method ade (inputan data berupa single data, terdiri dari semua column)
                process(x)

                #labeling as processed
                x.mark = "processed"
                db.add(x)
                db.commit()
                
                print(x)

    return {"status": "ok", "code": 200}


@app.get("/data/{data_id}", response_model=DummyResponse)
async def get_data_by_id(data_id:int):
   data=db.query(models.Data).filter(models.Data.id==data_id).first()

   if data is None:
      raise HTTPException(status_code=400, detail="data not found")

   return {"status": "ok", "code": 200, "data": data}


@app.delete("/data/{data_id}", response_model=DummyResponse)
async def delete_data(data_id:int):
   data_to_delete=db.query(models.Data).filter(models.Data.id==data_id).first()

   if data_to_delete is None:
      raise HTTPException(status_code=400, detail="tweet not found")

   db.delete(data_to_delete)
   db.commit()
   return {"status": "ok", "code": 200, "data":data_to_delete}

@app.put("/data/{data_id}", response_model=DummyResponse)
def labelling(data_id: int, category: str):
    tweet_to_update=db.query(models.Data).filter(models.Data.id==data_id).first()
    tweet_to_update.category = category
    
    
    db.commit()
    return {
        "status": "success",
        "code": 200,
        "messages": ["Tweet edited !"],
        "data": tweet_to_update,
    }

#: Start application
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5001)
