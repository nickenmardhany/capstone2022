from click import echo
from sqlalchemy.orm import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine=create_engine("postgresql://postgres:Akunicken99.@20.2.216.239:8000/capstone", echo = True)

Base = declarative_base()

SessionLocal = sessionmaker(bind=engine)
