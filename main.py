from __future__ import annotations

import base64
import csv
import hashlib
import hmac
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Optional, Type

import bcrypt
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
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

SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
ADMIN_ROLE = "ROLE_ADMIN"
DEFAULT_USER_ROLE = "ROLE_USER"
DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD = "admin123"
DEFAULT_ADMIN_ROLES = [ADMIN_ROLE]
security = HTTPBearer()


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


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    roles = Column(String, nullable=False, default=ADMIN_ROLE)


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


class UserResponse(ORMModel):
    id: int
    username: str
    roles: List[str]


class UserCreate(BaseModel):
    username: str
    password: str
    roles: Optional[List[str]] = None


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserDetailsResponse(BaseModel):
    id: int
    username: str
    roles: List[str]


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


def _serialize_roles(roles: List[str]) -> str:
    return ",".join(sorted(set(roles)))


def _parse_roles(raw_roles: str) -> List[str]:
    return [role for role in raw_roles.split(",") if role]


def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode(), salt).decode()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def _sign(message: bytes) -> str:
    signature = hmac.new(SECRET_KEY.encode(), message, hashlib.sha256).digest()
    return _b64url_encode(signature)


def encode_jwt(payload: dict) -> str:
    header = {"alg": ALGORITHM, "typ": "JWT"}
    header_b64 = _b64url_encode(json.dumps(header, separators=(",", ":")).encode())
    payload_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode())
    signing_input = f"{header_b64}.{payload_b64}".encode()
    signature_b64 = _sign(signing_input)
    return f"{header_b64}.{payload_b64}.{signature_b64}"


def decode_jwt(token: str) -> dict:
    try:
        header_b64, payload_b64, signature_b64 = token.split(".")
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token structure"
        )
    signing_input = f"{header_b64}.{payload_b64}".encode()
    expected_sig = _sign(signing_input)
    if not hmac.compare_digest(expected_sig, signature_b64):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token signature"
        )
    try:
        payload = json.loads(_b64url_decode(payload_b64))
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload"
        )

    exp = payload.get("exp")
    if exp is not None and int(time.time()) >= int(exp):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired"
        )
    return payload


def create_access_token(user: User) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user.id),
        "username": user.username,
        "roles": _parse_roles(user.roles),
        "iat": now,
        "exp": int(
            (now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)).timestamp()
        ),
    }
    # datetime objects are not JSON serializable by default
    payload["iat"] = int(now.timestamp())
    return encode_jwt(payload)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    token = credentials.credentials
    payload = decode_jwt(token)
    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload"
        )
    user = db.get(User, int(user_id))
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found"
        )
    return user


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    roles = _parse_roles(current_user.roles)
    if ADMIN_ROLE not in roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Admin privileges required"
        )
    return current_user


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


def ensure_default_admin() -> None:
    with SessionLocal() as session:
        exists = session.query(User).filter(User.username == DEFAULT_ADMIN_USERNAME).first()
        if not exists:
            admin = User(
                username=DEFAULT_ADMIN_USERNAME,
                hashed_password=hash_password(DEFAULT_ADMIN_PASSWORD),
                roles=_serialize_roles(DEFAULT_ADMIN_ROLES),
            )
            session.add(admin)
            session.commit()


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as session:
        has_data = session.query(Movie.movieId).first() is not None

    should_populate = os.environ.get("SKIP_DB_POPULATE") != "1"
    if not has_data and should_populate:
        populate_database()
    ensure_default_admin()


@app.on_event("startup")
def on_startup() -> None:
    init_db()


@app.get("/")
def read_root(current_user: User = Depends(get_current_user)) -> dict:
    return {"hello": "world"}


@app.post("/login", response_model=TokenResponse)
def login(data: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    user = db.query(User).filter(User.username == data.username).first()
    if user is None or not verify_password(data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials"
        )
    token = create_access_token(user)
    return TokenResponse(access_token=token)


@app.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def create_user(
    user: UserCreate, db: Session = Depends(get_db), admin: User = Depends(require_admin)
) -> User:
    if db.query(User).filter(User.username == user.username).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Username already exists"
        )
    roles = user.roles if user.roles is not None else [DEFAULT_USER_ROLE]
    new_user = User(
        username=user.username,
        hashed_password=hash_password(user.password),
        roles=_serialize_roles(roles),
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return UserResponse(
        id=new_user.id, username=new_user.username, roles=_parse_roles(new_user.roles)
    )


@app.get("/user_details", response_model=UserDetailsResponse)
def user_details(current_user: User = Depends(get_current_user)) -> UserDetailsResponse:
    return UserDetailsResponse(
        id=current_user.id,
        username=current_user.username,
        roles=_parse_roles(current_user.roles),
    )


@app.get("/movies", response_model=List[MovieResponse])
def get_movies(
    db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
) -> List[Movie]:
    return db.query(Movie).all()


@app.post("/movies", response_model=MovieResponse, status_code=status.HTTP_201_CREATED)
def create_movie(
    movie: MovieCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Movie:
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
def get_movie(
    movie_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Movie:
    return _get_or_404(db, Movie, movie_id, "Movie not found")


@app.put("/movies/{movie_id}", response_model=MovieResponse)
def update_movie(
    movie_id: int,
    movie_update: MovieUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Movie:
    movie = _get_or_404(db, Movie, movie_id, "Movie not found")
    movie.title = movie_update.title
    movie.genres = movie_update.genres
    db.commit()
    db.refresh(movie)
    return movie


@app.delete("/movies/{movie_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_movie(
    movie_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> None:
    movie = _get_or_404(db, Movie, movie_id, "Movie not found")
    db.delete(movie)
    db.commit()


@app.get("/links", response_model=List[LinkResponse])
def get_links(
    db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
) -> List[Link]:
    return db.query(Link).all()


@app.post("/links", response_model=LinkResponse, status_code=status.HTTP_201_CREATED)
def create_link(
    link: LinkCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Link:
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
def get_link(
    movie_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Link:
    return _get_or_404(db, Link, movie_id, "Link not found")


@app.put("/links/{movie_id}", response_model=LinkResponse)
def update_link(
    movie_id: int,
    link_update: LinkUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Link:
    link = _get_or_404(db, Link, movie_id, "Link not found")
    link.imdbId = link_update.imdbId
    link.tmdbId = link_update.tmdbId
    db.commit()
    db.refresh(link)
    return link


@app.delete("/links/{movie_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_link(
    movie_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> None:
    link = _get_or_404(db, Link, movie_id, "Link not found")
    db.delete(link)
    db.commit()


@app.get("/ratings", response_model=List[RatingResponse])
def get_ratings(
    db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
) -> List[Rating]:
    return db.query(Rating).all()


@app.post(
    "/ratings", response_model=RatingResponse, status_code=status.HTTP_201_CREATED
)
def create_rating(
    rating: RatingCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Rating:
    new_rating = Rating(**rating.model_dump())
    db.add(new_rating)
    db.commit()
    db.refresh(new_rating)
    return new_rating


@app.get("/ratings/{rating_id}", response_model=RatingResponse)
def get_rating(
    rating_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Rating:
    return _get_or_404(db, Rating, rating_id, "Rating not found")


@app.put("/ratings/{rating_id}", response_model=RatingResponse)
def update_rating(
    rating_id: int,
    rating_update: RatingUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
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
def delete_rating(
    rating_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> None:
    rating_obj = _get_or_404(db, Rating, rating_id, "Rating not found")
    db.delete(rating_obj)
    db.commit()


@app.get("/tags", response_model=List[TagResponse])
def get_tags(
    db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
) -> List[Tag]:
    return db.query(Tag).all()


@app.post("/tags", response_model=TagResponse, status_code=status.HTTP_201_CREATED)
def create_tag(
    tag: TagCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Tag:
    new_tag = Tag(**tag.model_dump())
    db.add(new_tag)
    db.commit()
    db.refresh(new_tag)
    return new_tag


@app.get("/tags/{tag_id}", response_model=TagResponse)
def get_tag(
    tag_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Tag:
    return _get_or_404(db, Tag, tag_id, "Tag not found")


@app.put("/tags/{tag_id}", response_model=TagResponse)
def update_tag(
    tag_id: int,
    tag_update: TagUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
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
def delete_tag(
    tag_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> None:
    tag_obj = _get_or_404(db, Tag, tag_id, "Tag not found")
    db.delete(tag_obj)
    db.commit()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
