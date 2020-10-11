"""loads a serialized model to make a prediction
"""
import typer
import pickle

load = __import__('utils').load
app = typer.Typer()


@app.command()
def predict(input: str = 'test.csv',
            modelfile: str = '/home/svelezg/Hands-on_ML/starter/mlruns/0/548e6aca1fe341e4a83d151a479e4066/artifacts/model/model.pkl'):
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
    print(data_pred[['Name', 'Survived']].head(20))

    total = data_pred.shape[0]
    survived = sum(data_pred['Survived'] == 1)
    percentage = round(100 * survived/total, 2)

    print(f"Survived: {survived}/{total} or {percentage}%")


if __name__ == "__main__":
    app()
