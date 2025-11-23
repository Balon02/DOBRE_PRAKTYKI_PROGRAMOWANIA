from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Optional

from fastapi import Depends, FastAPI
from pydantic import BaseModel, ConfigDict
from sqlalchemy import Column, Float, Integer, String, create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

app = FastAPI(title="MovieLens API")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "api_dbase"
DB_PATH = BASE_DIR / "movielens.sqlite"

DATABASE_URL = f"sqlite:///{DB_PATH}"
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

BULK_INSERT_SIZE = 5000


class Movie(Base):
    __tablename__ = "movies"

    movieId = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    genres = Column(String, nullable=False)


class Link(Base):
    __tablename__ = "links"

    movieId = Column(Integer, primary_key=True, index=True)
    imdbId = Column(String, nullable=True)
    tmdbId = Column(String, nullable=True)


class Rating(Base):
    __tablename__ = "ratings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    userId = Column(Integer, nullable=False, index=True)
    movieId = Column(Integer, nullable=False, index=True)
    rating = Column(Float, nullable=False)
    timestamp = Column(Integer, nullable=False)


class Tag(Base):
    __tablename__ = "tags"

    id = Column(Integer, primary_key=True, autoincrement=True)
    userId = Column(Integer, nullable=False, index=True)
    movieId = Column(Integer, nullable=False, index=True)
    tag = Column(String, nullable=True)
    timestamp = Column(Integer, nullable=False)


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class MovieResponse(ORMModel):
    movieId: int
    title: str
    genres: str


class LinkResponse(ORMModel):
    movieId: int
    imdbId: Optional[str] = None
    tmdbId: Optional[str] = None


class RatingResponse(ORMModel):
    id: int
    userId: int
    movieId: int
    rating: float
    timestamp: int


class TagResponse(ORMModel):
    id: int
    userId: int
    movieId: int
    tag: Optional[str] = None
    timestamp: int


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _optional_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    value = value.strip()
    return int(value) if value else None


def _optional_str(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _iterate_rows(
    filename: str, builder: Callable[[dict], Base]
) -> Iterator[Base]:
    csv_path = DATA_DIR / filename
    with csv_path.open(mode="r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            yield builder(row)


def _bulk_insert(session: Session, objects: Iterable[Base]) -> None:
    batch: List[Base] = []
    for obj in objects:
        batch.append(obj)
        if len(batch) >= BULK_INSERT_SIZE:
            session.bulk_save_objects(batch)
            session.commit()
            batch.clear()

    if batch:
        session.bulk_save_objects(batch)
        session.commit()


def _build_movie(row: dict) -> Movie:
    return Movie(
        movieId=int(row["movieId"]),
        title=row["title"],
        genres=row["genres"],
    )


def _build_link(row: dict) -> Link:
    return Link(
        movieId=int(row["movieId"]),
        imdbId=_optional_str(row.get("imdbId")),
        tmdbId=_optional_str(row.get("tmdbId")),
    )


def _build_rating(row: dict) -> Rating:
    return Rating(
        userId=int(row["userId"]),
        movieId=int(row["movieId"]),
        rating=float(row["rating"]),
        timestamp=int(row["timestamp"]),
    )


def _build_tag(row: dict) -> Tag:
    return Tag(
        userId=int(row["userId"]),
        movieId=int(row["movieId"]),
        tag=_optional_str(row.get("tag")),
        timestamp=int(row["timestamp"]),
    )


def populate_database() -> None:
    with SessionLocal() as session:
        _bulk_insert(session, _iterate_rows("movies.csv", _build_movie))
        _bulk_insert(session, _iterate_rows("links.csv", _build_link))
        _bulk_insert(session, _iterate_rows("ratings.csv", _build_rating))
        _bulk_insert(session, _iterate_rows("tags.csv", _build_tag))


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as session:
        has_data = session.query(Movie.movieId).first() is not None

    if not has_data:
        populate_database()


@app.on_event("startup")
def on_startup() -> None:
    init_db()


@app.get("/")
def read_root() -> dict:
    return {"hello": "world"}


@app.get("/movies", response_model=List[MovieResponse])
def get_movies(db: Session = Depends(get_db)) -> List[Movie]:
    return db.query(Movie).all()


@app.get("/links", response_model=List[LinkResponse])
def get_links(db: Session = Depends(get_db)) -> List[Link]:
    return db.query(Link).all()


@app.get("/ratings", response_model=List[RatingResponse])
def get_ratings(db: Session = Depends(get_db)) -> List[Rating]:
    return db.query(Rating).all()


@app.get("/tags", response_model=List[TagResponse])
def get_tags(db: Session = Depends(get_db)) -> List[Tag]:
    return db.query(Tag).all()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
