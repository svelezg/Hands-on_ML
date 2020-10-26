# ML API Service

## Authors
* **Solution:** Santiago Vélez G. [svelez.velezgarcia@gmail.com](svelez.velezgarcia@gmail.com) [@svelezg](https://github.com/svelezg)
* **Problem statement:** Cristian García



## Requirements
* Install Docker and Docker-compose

* Clone (download) the repository via 
``` 
    git clone https://github.com/svelezg/Hands-on_ML.git
```
* Navigate, cd into fastapi


## Usage

### API start
#### With docker compose
    sudo docker-compose up -d
    
#### With docker
    sudo docker build -t {name}api .
    sudo docker run -i --name {name}apicontainer -p 8000:8000 {name}api
    
    replace {name}
    
#### With Uvicorn
    python ./app/main.py


### Api testing

View it at [http://localhost:5000]([http://localhost:8000])

the server should respond with:

    {
    "message": "API live!"
    }

at the endpoint [http://0.0.0.0:8000/predict/](http://0.0.0.0:8000/predict/)
passing:
 
    {
    "Pclass": 3,  
    "Sex": "female",
    "Age": 25,
    "SibSp": 0,
    "Parch": 1,
    "Fare": 12,
    "Embarked": "S"
    }


the server should respond with output similar to:
```
    {
    "PassengerId": null,
    "Pclass": "1",
    "Name": "Paulo",
    "Sex": "male",
    "Age": 28.0,
    "SibSp": 0.0,
    "Parch": 0.0,
    "Ticket": 124124,
    "Fare": 80.0,
    "Cabin": null,
    "Embarked": "S",
    "Survived": 0
}
```



# Motivation
Machine Learning is software, as such its main purpose is to be used in a production environment which could be a server, mobile phone, web browser, IoT device, etc. Its very important for ML practitioners to have knowledge on how to deploy their own models.

# Goal
In this project we will be deploying our previously trained model on a web server and exposing it via a REST API. 

# Values
* **Scalability**: you should construct the architecture such that it can be scaled in the future.
* **Portability**: if possible your model should not be tied to a specific setup.

# Objectives
1. Expose a REST endpoint for your model. You should accept a JSON request and return a JSON with the prediction.
2. You should Dockerize your application to make it portable.
3. It should be easy to train a new model and deploy it, if possible it should be an automatic process.
4. Modify your previous project such that your training code makes deployment easier.
5. Your endpoint should be secured by an Authorization token.
6. (Bonus) Deploy it to a real production environment on a cloud service.
Create a new project that works on the MNIST dataset. This should help you make your template more project independent.

# Recommendations
Use FastAPI if possible, Flask or Django are also good.
Use docker-compose to test locally.


Example
```
Request

POST /api/titanic
Authorization: <TOKEN>

[{
    “PClass”: 1,
    “Sex”: “male”,
    ...
}]

Response

{
    “data”: [{
        “prediction”: 1,  # 0 or 1
        “score”: 0.98  # from 0 to 1
    }]
}
```