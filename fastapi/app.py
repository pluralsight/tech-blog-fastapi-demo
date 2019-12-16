import os
import pickle

from fastapi import FastAPI
import numpy as np
from aiohttp import ClientSession

from schemas import (
    RequestBody,
    ResponseBody,
    LabelResponseBody,
    ResponseValues,
    TextSample,
)


app = FastAPI(
    title="simple-model",
    description="a simple model-serving skateboard in FastAPI",
    version="0.1",
)


with open(os.getenv("MODEL_PATH"), "rb") as rf:
    clf = pickle.load(rf)

client_session = ClientSession()


@app.get("/healthcheck")
async def healthcheck():
    msg = (
        "this sentence is already halfway over, "
        "and still hasn't said anything at all"
    )
    return {"message": msg}


@app.post("/predict", response_model=ResponseBody)
async def predict(body: RequestBody):
    data = np.array(body.to_array())

    probas = clf.predict_proba(data)
    predictions = probas.argmax(axis=1)

    return {
        "predictions": (
            np.tile(clf.classes_, (len(predictions), 1))[
                np.arange(len(predictions)), predictions
            ].tolist()
        ),
        "probabilities": probas[np.arange(len(predictions)), predictions].tolist(),
    }


@app.post("/predict/{label}", response_model=LabelResponseBody)
async def predict_label(label: ResponseValues, body: RequestBody):
    data = np.array(body.to_array())

    probas = clf.predict_proba(data)
    target_idx = clf.classes_.tolist().index(label.value)

    return {"label": label.value, "probabilities": probas[:, target_idx].tolist()}


@app.get("/cat-facts", response_model=TextSample)
async def cat_facts():
    url = "https://cat-fact.herokuapp.com/facts/random"
    async with client_session.get(url) as resp:
        response = await resp.json()

    return response


@app.on_event("shutdown")
async def cleanup():
    await client_session.close()
