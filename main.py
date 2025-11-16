from __future__ import annotations

import csv
from dataclasses import dataclass, fields
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Type, TypeVar

from fastapi import FastAPI

app = FastAPI(title="MovieLens API")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "api_dbase"

T = TypeVar("T")


@dataclass
class Movie:
    movieId: int
    title: str
    genres: str


@dataclass
class Link:
    movieId: int
    imdbId: str
    tmdbId: str


@dataclass
class Rating:
    userId: int
    movieId: int
    rating: float
    timestamp: int


@dataclass
class Tag:
    userId: int
    movieId: int
    tag: str
    timestamp: int


def _convert_value(value: str, target_type: Type) -> object:
    if value == "" or value is None:
        return None
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    return value


def _deserialize(row: Dict[str, str], model_cls: Type[T]) -> T:
    kwargs = {
        field.name: _convert_value(row.get(field.name), field.type)
        for field in fields(model_cls)
    }
    return model_cls(**kwargs)


def _load_dataset(filename: str, model_cls: Type[T]) -> List[Dict]:
    csv_path = DATA_DIR / filename
    with csv_path.open(mode="r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        items = [_deserialize(row, model_cls).__dict__ for row in reader]
    return items


@lru_cache(maxsize=1)
def load_movies() -> List[Dict]:
    return _load_dataset("movies.csv", Movie)


@lru_cache(maxsize=1)
def load_links() -> List[Dict]:
    return _load_dataset("links.csv", Link)


@lru_cache(maxsize=1)
def load_ratings() -> List[Dict]:
    return _load_dataset("ratings.csv", Rating)


@lru_cache(maxsize=1)
def load_tags() -> List[Dict]:
    return _load_dataset("tags.csv", Tag)


@app.get("/")
def read_root() -> Dict[str, str]:
    return {"hello": "world"}


@app.get("/movies")
def get_movies() -> List[Dict]:
    return load_movies()


@app.get("/links")
def get_links() -> List[Dict]:
    return load_links()


@app.get("/ratings")
def get_ratings() -> List[Dict]:
    return load_ratings()


@app.get("/tags")
def get_tags() -> List[Dict]:
    return load_tags()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
