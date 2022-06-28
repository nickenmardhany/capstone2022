from database import Base, engine
from models import User

print("Create database....")

Base.metadata.create_all(engine)