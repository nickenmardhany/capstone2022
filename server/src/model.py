import re
import string
import nltk
import pickle
import pandas as pd
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from pydantic import BaseModel

from database import SessionLocal
import models


nltk.download('punkt')

path = 'pipeline.pkl'
clf1 = pickle.load(open(path, 'rb'))


data1 = pd.read_csv("test.csv", encoding='latin1')
data1.columns = ["Label", "Tweet"]

class Dummy2(BaseModel):
    id: int
    tweets: str
    users: str
    mark : str
    label: str
    type: str

    class Config:
            orm_mode=True

db=SessionLocal()

Data: list[Dummy2] = []

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


def predict_data():

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

predict_data()

