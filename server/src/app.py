from typing import List
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

class Dummy2(BaseModel):
    id: int
    tweets: str
    users: str
    mark : str
    label: str
    type: str

    class Config:
            orm_mode=True

class User(BaseModel):
    id: str
    first_name: str
    last_name: str
    email: str
    class Config:
        orm_mode=True

db=SessionLocal()

Data: list[Dummy2] = []


#: Describe all Pydantic Response classes
class ResponseBase(BaseModel):
    status: str
    code: int
    messages: List[str] = []

class DummyResponse(ResponseBase):
    data: Dummy2

class TweetsResponse(ResponseBase):
    data: Tweets

class ListTweetsResponse(ResponseBase):
    data: List[Tweets]

class ListDummyResponse(ResponseBase):
    data: List[Dummy2]

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
                new_tweet=models.Dummy2(
                
                users = json_response['includes']['users'][0]['username'],
                tweets=json_response['data']['text'],
                mark='unprocessed',
                label='bukan pengaduan',
                type='belum ditentukan'
                )
        
                
                db.add(new_tweet)
                db.commit()
            # print(json.dumps(tweets.decode("utf-8"), indent=4, sort_keys=True))
            print(json.dumps(json_response, indent=4, sort_keys=True))

    return {"status": "ok", "code": 200, "data": tweets.decode("utf-8")} 

#MODEL

#Preprocessing 

def clean1(isi):
##URl
  isi = re.sub(r"htt\S{0,}", " ",isi)
##USER
  isi = re.sub(r'@\S{0,}', ' USER ',isi)
  isi = re.sub(r"\s+", " ", isi)
##LOWER
  isi = isi.lower()
##ANGKA
  isi = re.sub(r"\d+", "", isi)
#TANDA BACA
  isi = isi.translate(str.maketrans("","",string.punctuation))
##stop(isi):
  factorystop = StopWordRemoverFactory()
  stop = factorystop.create_stop_word_remover()
  isi = stop.remove(isi)
  isi = re.sub('[^a-zA-Z0-9\n\.]', ' ', isi)
  factorystem = StemmerFactory()
  stemmer = factorystem.create_stemmer()
  isi = stemmer.stem(isi)

  return isi



#method ade if output = 1, add to database pengaduan

def process(data: Dummy2):
    
      data.tweets = [clean1(data.tweets)]
      result = clf1.predict(data.tweets)
      if result == 1:
        data.label = "pengaduan"
      else:
        data.label = "bukan pengaduan"

      db.add(data)
      db.commit()
      
      data2=db.query(models.Dummy2).all()
      print(data2)
      return { 
          "status": "success",
          "code": 200,
          "messages": ["Data processed!"],
          "data": data2
          }

@app.get("/data", response_model=ListDummyResponse)
async def get_all_data():
   tweet=db.query(models.Dummy2).all()
   return {"status": "ok", "code": 200, "data": tweet}


@app.get("/data/pengaduan", response_model=ListDummyResponse)
async def get_data_pengaduan():
   data=db.query(models.Dummy2).filter(models.Dummy2.label=='pengaduan').all()
   return {"status": "ok", "code": 200, "data": data}


@app.post("/data", status_code=201, response_model=DummyResponse)
def create_data_pengaduan(user: Dummy2):
    new_user=models.Dummy2( id = user.id, tweets=user.tweets,users=user.users,mark=user.mark,label= user.label,type = user.type)

    db.add(new_user)
    db.commit()
    return {
        "status": "success",
        "code": 201,
        "messages": ["Book added !"],
        "data": new_user,
    }

@app.get("/predict_data", response_model=ListDummyResponse)
async def predict_laporan():
    data=db.query(models.Dummy2).all()
      
    for x in data:
            if x.mark == "unprocessed":
                #panggil method ade (inputan data berupa single data, terdiri dari semua column)
                process(x)

                #labeling as processed
                x.mark = "processed"
                db.add(x)
                db.commit()
                
                print(x)

    return {"status": "ok", "code": 200, "data": Data}


@app.get("/data/{data_id}", response_model=DummyResponse)
async def get_data_by_id(data_id:int):
   data=db.query(models.Dummy2).filter(models.Dummy2.id==data_id).first()

   if data is None:
      raise HTTPException(status_code=400, detail="data not found")

   return {"status": "ok", "code": 200, "data": data}


@app.delete("/data/{data_id}", response_model=DummyResponse)
async def delete_data(data_id:int):
   data_to_delete=db.query(models.Dummy2).filter(models.Dummy2.id==data_id).first()

   if data_to_delete is None:
      raise HTTPException(status_code=400, detail="tweet not found")

   db.delete(data_to_delete)
   db.commit()
   return {"status": "ok", "code": 200, "data":data_to_delete}

@app.put("/data/{data_id}", response_model=DummyResponse)
def labelling(data_id: int, type: str):
    tweet_to_update=db.query(models.Dummy2).filter(models.Dummy2.id==data_id).first()
    tweet_to_update.type = type
    
    
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
