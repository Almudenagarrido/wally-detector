import os
import glob
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import mlflow.sklearn

IMG_SIZE = (32, 32)

def load_images(folder, label):
    features = []
    labels = []
    for path in glob.glob(os.path.join(folder, "*.jpg")):
        img = Image.open(path).convert("RGB").resize(IMG_SIZE)
        arr = np.array(img).flatten() / 255.0
        features.append(arr)
        labels.append(label)
    return features, labels

wally_X, wally_y = load_images("data/wally", 1)
no_wally_X, no_wally_y = load_images("data/no_wally", 0)

X = np.array(wally_X + no_wally_X)
y = np.array(wally_y + no_wally_y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

with mlflow.start_run():
    n_estimators = 100
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    joblib.dump(model, "model.pkl")
    mlflow.sklearn.log_model(model, "wally-detector-model")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("accuracy", accuracy)

    print(f"Modelo entrenado y precisión: {accuracy:.4f}")
    print("Experimento registrado con MLflow.")
