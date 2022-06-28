
from banal import first
from database import Base
from sqlalchemy import String, Boolean, Integer, Column, null
from typing import Optional, List, Text
from uuid import UUID, uuid4
from pydantic import BaseModel
from enum import Enum

class User(Base):
    __tablename__='user'
    id=Column(Integer, primary_key= True)
    user_name=Column(String(255),unique= False,nullable= False)
    password=Column(String(255), unique= False, nullable= False)
    first_name=Column(String(255),unique= False,nullable= False)
    last_name=Column(String(255),unique= False,nullable= False)
    email=Column(String(255),unique= True,nullable= False)


    def __repr__(self):
        return f"<User first_name ={self.first_name}>"

class Tweets(Base):
    __tablename__='tweets'
    id=Column(Integer, primary_key= True)
    tweets=Column(String(1000),unique= False,nullable= False)

class Dummy2(Base):
    __tablename__='dummy2'
    id=Column(Integer, primary_key= True)
    users=Column(String(255),unique= False,nullable= False)
    tweets=Column(String(1000),unique= False,nullable= False)
    mark=Column(String(255),unique= False,nullable= False)
    label=Column(String(255),unique= False,nullable= False)
    type=Column(String(255),unique= False,nullable= False)

    
    
    
    

    def __repr__(self):
        return f"<tweets tweets ={self.tweets}>"