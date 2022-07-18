from pydantic import BaseModel


class AuthDetails(BaseModel):
    first_name: str
    last_name: str
    email: str
    username: str
    password: str

    class Config:
            orm_mode=True

class Login(BaseModel):
    email: str
    password: str

    class Config:
            orm_mode=True
    