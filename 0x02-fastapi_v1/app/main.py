"""
Exposes a secured REST endpoint for the model
loads a serialized model to make a prediction
"""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
import uvicorn
import os

import pickle5 as pickle
import pandas as pd

# to get a string like this run:
# openssl rand -hex 32
SECRET_KEY = "673860d8e2814b9bee205d6db61d05dd6590dbd78d7b774e91c2d2280adde270"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

fake_users_db = {
    "johndoe": {
        "username": "johndoe",
        "full_name": "John Doe",
        "email": "johndoe@example.com",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",
        "disabled": False,
    },
    "alice": {
        "username": "alice",
        "full_name": "Alice Wonderson",
        "email": "alice@example.com",
        "hashed_password": "$2y$12$hTa7DRRH8OK7GFHuZByaZuIvEtM15k7w3yoV05LbrIgJ2tKNPYjae",
        "disabled": True,
    },
    "svelezg": {
        "username": "svelezg",
        "full_name": "Santiago Vélez",
        "email": "svelez@example.com",
        "hashed_password": "$2y$12$ITPuzyLGHD4TdxLeVJQuAuYOPJyZDx7mhqsJrCE/JH6jhAErmU5MW",
        "disabled": False,
    },
}


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None


class UserInDB(User):
    hashed_password: str


class IncomingData(BaseModel):
    PassengerId: Optional[str] = None
    Pclass: str
    Name: Optional[str] = None
    Sex: str
    Age: float
    SibSp: float
    Parch: float
    Ticket: Optional[int] = None
    Fare: float
    Cabin: Optional[str] = None
    Embarked: Optional[str] = None


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI()


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    print(pwd_context.hash(password))
    return pwd_context.hash(password)


def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)


def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/users/me/", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user


@app.get("/users/me/items/")
async def read_own_items(current_user: User = Depends(get_current_active_user)):
    return [{"item_id": "Foo", "owner": current_user.username}]


# Api root or home endpoint
@app.get('/secure')
@app.get('/secure/home')
def read_home(current_user: User = Depends(get_current_active_user)):
    """
    Home endpoint which can be used to test the availability
    of the application.
    :return: Dict with key 'message' and value 'API live and secured!'
    """
    return {'message': 'API live and secured!'}


# Api root or home endpoint
@app.get('/')
@app.get('/home')
def read_home():
    """
    Home endpoint which can be used to test the availability
    of the application.
    :return: Dict with key 'message' and value 'API live!'
    """
    return {'message': 'API live!'}


# Prediction endpoint
@app.put("/predict")
async def predict(incoming_data: IncomingData, current_user: User = Depends(get_current_active_user)):
    # load data and convert to pandas dataframe
    my_dict = incoming_data.dict()
    df = pd.DataFrame([my_dict])

    # load model from pickle file
    modelfile = 'app/model.pkl'
    file = open(modelfile, 'rb')
    model = pickle.load(file)
    file.close()

    # make the prediction
    y_pred = model.predict(df)
    survived = int(y_pred[0, 0])

    # create results dict
    result = my_dict
    result.update({"Survived": survived})
    return result


if __name__ == "__main__":
    # Run app with uvicorn with port and host specified.
    # Host needed for docker port mapping
    uvicorn.run(app, port=int(os.environ.get('PORT', 8000)), host="0.0.0.0")
