import argparse
import pickle

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def train_model(pickle_file_path: str, seed: int = 42) -> None:
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

    print(f"train score: {pipe.score(X_train, y_train):.3f}")
    print(f"test score: {pipe.score(X_test, y_test):.3f}")

    with open(pickle_file_path, "wb") as wf:
        pickle.dump(pipe, wf)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        "-f",
        default="./model.pkl",
        help="target filepath to write model pickle file",
    )
    parser.add_argument(
        "--seed",
        "-s",
        default=42,
        type=int,
        help="random seed for train/test split and model training",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    train_model(args.file, args.seed)
