"""
Exposes a REST endpoint for the model
loads a serialized model to make a prediction
"""

from typing import Optional
import pickle5 as pickle
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import uvicorn


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


app = FastAPI()


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
async def predict(incoming_data: IncomingData):
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
    uvicorn.run(app, port=8000, host="0.0.0.0")
