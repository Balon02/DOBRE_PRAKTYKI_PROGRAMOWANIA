from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Optional, Type

from fastapi import Depends, FastAPI, HTTPException, status
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


class MovieCreate(BaseModel):
    movieId: int
    title: str
    genres: str


class MovieUpdate(BaseModel):
    title: str
    genres: str


class LinkResponse(ORMModel):
    movieId: int
    imdbId: Optional[str] = None
    tmdbId: Optional[str] = None


class LinkCreate(BaseModel):
    movieId: int
    imdbId: Optional[str] = None
    tmdbId: Optional[str] = None


class LinkUpdate(BaseModel):
    imdbId: Optional[str] = None
    tmdbId: Optional[str] = None


class RatingResponse(ORMModel):
    id: int
    userId: int
    movieId: int
    rating: float
    timestamp: int


class RatingCreate(BaseModel):
    userId: int
    movieId: int
    rating: float
    timestamp: int


class RatingUpdate(BaseModel):
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


class TagCreate(BaseModel):
    userId: int
    movieId: int
    tag: Optional[str] = None
    timestamp: int


class TagUpdate(BaseModel):
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


def _get_or_404(
    db: Session, model: Type[Base], pk: int, message: str
) -> Base:
    instance = db.get(model, pk)
    if instance is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=message)
    return instance


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


@app.post("/movies", response_model=MovieResponse, status_code=status.HTTP_201_CREATED)
def create_movie(movie: MovieCreate, db: Session = Depends(get_db)) -> Movie:
    if db.get(Movie, movie.movieId):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Movie already exists"
        )
    new_movie = Movie(**movie.model_dump())
    db.add(new_movie)
    db.commit()
    db.refresh(new_movie)
    return new_movie


@app.get("/movies/{movie_id}", response_model=MovieResponse)
def get_movie(movie_id: int, db: Session = Depends(get_db)) -> Movie:
    return _get_or_404(db, Movie, movie_id, "Movie not found")


@app.put("/movies/{movie_id}", response_model=MovieResponse)
def update_movie(
    movie_id: int, movie_update: MovieUpdate, db: Session = Depends(get_db)
) -> Movie:
    movie = _get_or_404(db, Movie, movie_id, "Movie not found")
    movie.title = movie_update.title
    movie.genres = movie_update.genres
    db.commit()
    db.refresh(movie)
    return movie


@app.delete("/movies/{movie_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_movie(movie_id: int, db: Session = Depends(get_db)) -> None:
    movie = _get_or_404(db, Movie, movie_id, "Movie not found")
    db.delete(movie)
    db.commit()


@app.get("/links", response_model=List[LinkResponse])
def get_links(db: Session = Depends(get_db)) -> List[Link]:
    return db.query(Link).all()


@app.post("/links", response_model=LinkResponse, status_code=status.HTTP_201_CREATED)
def create_link(link: LinkCreate, db: Session = Depends(get_db)) -> Link:
    if db.get(Link, link.movieId):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Link already exists"
        )
    new_link = Link(**link.model_dump())
    db.add(new_link)
    db.commit()
    db.refresh(new_link)
    return new_link


@app.get("/links/{movie_id}", response_model=LinkResponse)
def get_link(movie_id: int, db: Session = Depends(get_db)) -> Link:
    return _get_or_404(db, Link, movie_id, "Link not found")


@app.put("/links/{movie_id}", response_model=LinkResponse)
def update_link(
    movie_id: int, link_update: LinkUpdate, db: Session = Depends(get_db)
) -> Link:
    link = _get_or_404(db, Link, movie_id, "Link not found")
    link.imdbId = link_update.imdbId
    link.tmdbId = link_update.tmdbId
    db.commit()
    db.refresh(link)
    return link


@app.delete("/links/{movie_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_link(movie_id: int, db: Session = Depends(get_db)) -> None:
    link = _get_or_404(db, Link, movie_id, "Link not found")
    db.delete(link)
    db.commit()


@app.get("/ratings", response_model=List[RatingResponse])
def get_ratings(db: Session = Depends(get_db)) -> List[Rating]:
    return db.query(Rating).all()


@app.post(
    "/ratings", response_model=RatingResponse, status_code=status.HTTP_201_CREATED
)
def create_rating(rating: RatingCreate, db: Session = Depends(get_db)) -> Rating:
    new_rating = Rating(**rating.model_dump())
    db.add(new_rating)
    db.commit()
    db.refresh(new_rating)
    return new_rating


@app.get("/ratings/{rating_id}", response_model=RatingResponse)
def get_rating(rating_id: int, db: Session = Depends(get_db)) -> Rating:
    return _get_or_404(db, Rating, rating_id, "Rating not found")


@app.put("/ratings/{rating_id}", response_model=RatingResponse)
def update_rating(
    rating_id: int, rating_update: RatingUpdate, db: Session = Depends(get_db)
) -> Rating:
    rating_obj = _get_or_404(db, Rating, rating_id, "Rating not found")
    rating_obj.userId = rating_update.userId
    rating_obj.movieId = rating_update.movieId
    rating_obj.rating = rating_update.rating
    rating_obj.timestamp = rating_update.timestamp
    db.commit()
    db.refresh(rating_obj)
    return rating_obj


@app.delete("/ratings/{rating_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_rating(rating_id: int, db: Session = Depends(get_db)) -> None:
    rating_obj = _get_or_404(db, Rating, rating_id, "Rating not found")
    db.delete(rating_obj)
    db.commit()


@app.get("/tags", response_model=List[TagResponse])
def get_tags(db: Session = Depends(get_db)) -> List[Tag]:
    return db.query(Tag).all()


@app.post("/tags", response_model=TagResponse, status_code=status.HTTP_201_CREATED)
def create_tag(tag: TagCreate, db: Session = Depends(get_db)) -> Tag:
    new_tag = Tag(**tag.model_dump())
    db.add(new_tag)
    db.commit()
    db.refresh(new_tag)
    return new_tag


@app.get("/tags/{tag_id}", response_model=TagResponse)
def get_tag(tag_id: int, db: Session = Depends(get_db)) -> Tag:
    return _get_or_404(db, Tag, tag_id, "Tag not found")


@app.put("/tags/{tag_id}", response_model=TagResponse)
def update_tag(
    tag_id: int, tag_update: TagUpdate, db: Session = Depends(get_db)
) -> Tag:
    tag_obj = _get_or_404(db, Tag, tag_id, "Tag not found")
    tag_obj.userId = tag_update.userId
    tag_obj.movieId = tag_update.movieId
    tag_obj.tag = tag_update.tag
    tag_obj.timestamp = tag_update.timestamp
    db.commit()
    db.refresh(tag_obj)
    return tag_obj


@app.delete("/tags/{tag_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_tag(tag_id: int, db: Session = Depends(get_db)) -> None:
    tag_obj = _get_or_404(db, Tag, tag_id, "Tag not found")
    db.delete(tag_obj)
    db.commit()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
