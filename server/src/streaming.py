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


class Tweets(BaseModel):
    id: int
    tweets: str
    class Config:
            orm_mode=True

class Data(BaseModel):
    id: int
    tweets: str
    users: str
    mark : str
    label: str
    type: str

    class Config:
            orm_mode=True

db=SessionLocal()

Data: list[Data] = []

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
                user = json_response['includes']['users'][0]['username'],
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


#: Start application
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
