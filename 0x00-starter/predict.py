"""loads a serialized model to make a prediction
"""
import typer
import pickle

load = __import__('utils').load
app = typer.Typer()


@app.command()
def predict(input: str = 'test.csv',
            modelfile: str = '/home/svelezg/Hands-on_ML/starter/mlruns/0/80b3caa0c58b47e68483ff13ef619446/artifacts/model/model.pkl'):
    data = load(input)

    # open a file, where you stored the pickled data
    file = open(modelfile, 'rb')

    # load the model
    model = pickle.load(file)

    # close the file
    file.close()

    # make the prediction
    y_pred = model.predict(data)
    data_pred = data.copy()
    data_pred['Survived'] = y_pred
    print(data_pred[['Age', 'Pclass', 'Sex', 'Fare', 'Survived']].head(20).T)

    total = data_pred.shape[0]
    survived = sum(data_pred['Survived'] == 1)
    percentage = round(100 * survived / total, 2)

    print(f"Survived: {survived}/{total} or {percentage}%")


if __name__ == "__main__":
    app()
