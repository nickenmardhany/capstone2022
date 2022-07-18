from click import echo
from sqlalchemy.orm import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine=create_engine("postgresql://postgres:Akunicken99.@127.0.0.1:5432/capstone2", echo = True)

Base = declarative_base()

SessionLocal = sessionmaker(bind=engine)
