import os
import pickle

from flask import Flask, jsonify, request
import numpy as np


app = Flask(__name__)

with open(os.getenv("MODEL_PATH"), "rb") as rf:
    clf = pickle.load(rf)


@app.route("/healthcheck", methods=["GET"])
def healthcheck():
    msg = (
        "this sentence is already halfway over, "
        "and still hasn't said anything at all"
    )
    return jsonify({"message": msg})


@app.route("/predict", methods=["POST"])
def predict():
    samples = request.get_json()["samples"]
    data = np.array([sample["text"] for sample in samples])

    probas = clf.predict_proba(data)
    predictions = probas.argmax(axis=1)

    return jsonify(
        {
            "predictions": (
                np.tile(clf.classes_, (len(predictions), 1))[
                    np.arange(len(predictions)), predictions
                ].tolist()
            ),
            "probabilities": probas[np.arange(len(predictions)), predictions].tolist(),
        }
    )


@app.route("/predict/<label>", methods=["POST"])
def predict_label(label):
    samples = request.get_json()["samples"]
    data = np.array([sample["text"] for sample in samples])

    probas = clf.predict_proba(data)
    target_idx = clf.classes_.tolist().index(label)

    return jsonify({"label": label, "probabilities": probas[:, target_idx].tolist()})
