import pickle

import typer

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def train_model(
    pickle_file_path: str = typer.Option(
        "./model.pkl",
        "--file",
        "-f",
        help="filepath to write pickle file",
        show_default=True,
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        "-s",
        help="random-number seed passed to scikit-learn",
        show_default=True,
    ),
) -> None:
    """pulls a subset of the "20 Newsgroups" dataset and builds a simple
    Naive Bayes classifier.
    """
    data = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes"),
        categories=["rec.sport.hockey", "sci.space", "talk.politics.misc"],
    )
    X = [d.lower() for d in data.data]
    y = np.tile(data.target_names, (len(data.target), 1))[
        np.arange(len(data.target)), data.target
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed
    )

    pipe = Pipeline(
        [
            ("vectorizer", CountVectorizer(stop_words="english")),
            ("tfidf", TfidfTransformer()),
            ("clf", MultinomialNB()),
        ]
    )
    pipe.fit(X_train, y_train)

    typer.echo(f"train score: {pipe.score(X_train, y_train):.3f}")
    typer.echo(f"test score: {pipe.score(X_test, y_test):.3f}")

    with open(pickle_file_path, "wb") as wf:
        pickle.dump(pipe, wf)


if __name__ == "__main__":
    typer.run(train_model)
