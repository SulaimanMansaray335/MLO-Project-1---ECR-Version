import os 
import sys 
import numpy as np 
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from importlib import resources
import pkgutil
from pathlib import Path

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok = True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_models(x_train, y_train, x_test, y_test, models, params):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]

            
            gs = GridSearchCV(model, param, cv = 3)
            gs.fit(x_train, y_train)
            #model.fit(x_train, y_train) #train model

            model.set_params(**gs.best_params_)
            model.fit(x_train, y_train)

            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

            return report 

    except Exception as e:
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        # If given a real filesystem path, open it normally.
        if os.path.exists(file_path):
            with open(file_path, "rb") as file_obj:
                return dill.load(file_obj)

        # Otherwise, attempt to load the file as a package resource.
        # Accept paths like "artifacts/model.pkl" or just "model.pkl".
        resource_path = Path(file_path)
        parts = list(resource_path.parts)

        # Try importlib.resources (Python 3.9+)
        try:
            pkg = __package__ or 'src'
            res = resources.files(pkg).joinpath(*parts)
            with res.open('rb') as file_obj:
                return dill.load(file_obj)
        except Exception:
            # Fallback to pkgutil.get_data which returns bytes
            try:
                pkg = __package__ or 'src'
                data = pkgutil.get_data(pkg, "/".join(parts))
                if data is None:
                    raise FileNotFoundError(f"Resource not found: {file_path}")
                return dill.loads(data)
            except Exception as e:
                raise

    except Exception as e:
        raise CustomException(e, sys)