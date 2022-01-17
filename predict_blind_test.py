import os

from src.utils.preprocessing import cup_create_df

def predict_btest(ts_path, model):
    ts_path = os.path.join(os.path.dirname(__file__), ts_path)
    test_df = cup_create_df(ts_path, False)
    test_df = test_df.drop(columns="id", axis=1, inplace=False)
    X_test = test_df.to_numpy()
    y_pred = model.predict(X_test)
    names = "Name1  Surname1	Name2 Surname2" #TODO
    filename = "cup_res.out" # TODO
    teamname = "ML's Angel" #TODO
    title = "ML-CUP21"
    date = "22/02/2021" #TODO
    with open(filename, "w") as fd:
        fd.write("{:s}\n".format(names))
        fd.write("{:s}\n".format(teamname))
        fd.write("{:s}\n".format(title))
        fd.write("{:s}\n".format(date))
        for i in range(y_pred.shape[0]):
            fd.write("{:d},{:f},{:f}\n".format(i+1, y_pred[i,0], y_pred[i,1]))
    print(y_pred)
    exit(0)