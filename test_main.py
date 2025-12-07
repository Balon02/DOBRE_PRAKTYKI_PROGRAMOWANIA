import pytest
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

import main


def seed_database(session: Session) -> None:
    movies = [
        main.Movie(movieId=1, title="Toy Story (1995)", genres="Adventure|Animation"),
        main.Movie(movieId=2, title="Jumanji (1995)", genres="Adventure"),
    ]
    links = [
        main.Link(movieId=1, imdbId="114709", tmdbId="862"),
        main.Link(movieId=2, imdbId="113497", tmdbId="8844"),
    ]
    ratings = [
        main.Rating(userId=1, movieId=1, rating=4.5, timestamp=964982703),
        main.Rating(userId=2, movieId=2, rating=3.0, timestamp=964982703),
    ]
    tags = [
        main.Tag(userId=1, movieId=1, tag="funny", timestamp=964982703),
        main.Tag(userId=2, movieId=2, tag="board game", timestamp=964982703),
    ]
    session.add_all(movies + links + ratings + tags)
    session.commit()


@pytest.fixture()
def client(tmp_path):
    db_path = tmp_path / "test.sqlite"
    test_engine = create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
    )
    TestingSessionLocal = sessionmaker(
        bind=test_engine, autoflush=False, autocommit=False
    )

    main.engine = test_engine
    main.SessionLocal = TestingSessionLocal
    main.Base.metadata.drop_all(bind=test_engine)
    main.Base.metadata.create_all(bind=test_engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    main.app.dependency_overrides[main.get_db] = override_get_db

    with TestingSessionLocal() as session:
        seed_database(session)

    with TestClient(main.app) as test_client:
        yield test_client

    main.app.dependency_overrides.clear()


def test_get_movies_returns_all(client: TestClient):
    response = client.get("/movies")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert len(data) == 2
    assert data[0]["title"] == "Toy Story (1995)"


def test_get_movie_by_id_returns_item(client: TestClient):
    response = client.get("/movies/1")
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["movieId"] == 1


def test_get_movie_by_id_missing_returns_404(client: TestClient):
    response = client.get("/movies/999")
    assert response.status_code == status.HTTP_404_NOT_FOUND


def test_create_movie_adds_record(client: TestClient):
    payload = {"movieId": 3, "title": "Apollo 13 (1995)", "genres": "Drama"}
    response = client.post("/movies", json=payload)
    assert response.status_code == status.HTTP_201_CREATED
    created = response.json()
    assert created["movieId"] == payload["movieId"]
    assert created["title"] == payload["title"]

    list_response = client.get("/movies")
    assert len(list_response.json()) == 3


def test_update_movie_persists_changes(client: TestClient):
    payload = {"title": "Toy Story (Updated)", "genres": "Adventure"}
    response = client.put("/movies/1", json=payload)
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["title"] == payload["title"]

    verify_response = client.get("/movies/1")
    assert verify_response.json()["genres"] == payload["genres"]


def test_delete_movie_removes_record(client: TestClient):
    response = client.delete("/movies/2")
    assert response.status_code == status.HTTP_204_NO_CONTENT

    missing_response = client.get("/movies/2")
    assert missing_response.status_code == status.HTTP_404_NOT_FOUND


def test_get_link_by_id_returns_item(client: TestClient):
    response = client.get("/links/1")
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["tmdbId"] == "862"


def test_get_link_missing_returns_404(client: TestClient):
    response = client.get("/links/999")
    assert response.status_code == status.HTTP_404_NOT_FOUND


def test_create_link_adds_record(client: TestClient):
    payload = {"movieId": 3, "imdbId": "94943", "tmdbId": "9480"}
    response = client.post("/links", json=payload)
    assert response.status_code == status.HTTP_201_CREATED
    assert response.json()["movieId"] == payload["movieId"]

    verify_response = client.get("/links/3")
    assert verify_response.status_code == status.HTTP_200_OK


def test_update_link_persists_changes(client: TestClient):
    payload = {"imdbId": "114709-ALT", "tmdbId": "862"}
    response = client.put("/links/1", json=payload)
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["imdbId"] == payload["imdbId"]


def test_delete_link_removes_record(client: TestClient):
    response = client.delete("/links/2")
    assert response.status_code == status.HTTP_204_NO_CONTENT
    verify_response = client.get("/links/2")
    assert verify_response.status_code == status.HTTP_404_NOT_FOUND


def test_get_ratings_returns_all(client: TestClient):
    response = client.get("/ratings")
    assert response.status_code == status.HTTP_200_OK
    assert len(response.json()) == 2


def test_get_rating_by_id_returns_item(client: TestClient):
    response = client.get("/ratings/1")
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["rating"] == 4.5


def test_get_rating_missing_returns_404(client: TestClient):
    response = client.get("/ratings/999")
    assert response.status_code == status.HTTP_404_NOT_FOUND


def test_create_rating_adds_record(client: TestClient):
    payload = {"userId": 3, "movieId": 1, "rating": 5.0, "timestamp": 111111}
    response = client.post("/ratings", json=payload)
    assert response.status_code == status.HTTP_201_CREATED
    created = response.json()
    assert created["userId"] == payload["userId"]
    assert created["rating"] == payload["rating"]


def test_update_rating_changes_values(client: TestClient):
    payload = {"userId": 1, "movieId": 1, "rating": 2.5, "timestamp": 222222}
    response = client.put("/ratings/1", json=payload)
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["rating"] == payload["rating"]

    verify_response = client.get("/ratings/1")
    assert verify_response.json()["timestamp"] == payload["timestamp"]


def test_delete_rating_removes_record(client: TestClient):
    response = client.delete("/ratings/2")
    assert response.status_code == status.HTTP_204_NO_CONTENT
    verify_response = client.get("/ratings/2")
    assert verify_response.status_code == status.HTTP_404_NOT_FOUND


def test_get_tags_returns_all(client: TestClient):
    response = client.get("/tags")
    assert response.status_code == status.HTTP_200_OK
    assert len(response.json()) == 2


def test_get_tag_by_id_returns_item(client: TestClient):
    response = client.get("/tags/1")
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["tag"] == "funny"


def test_get_tag_missing_returns_404(client: TestClient):
    response = client.get("/tags/999")
    assert response.status_code == status.HTTP_404_NOT_FOUND


def test_create_tag_adds_record(client: TestClient):
    payload = {"userId": 3, "movieId": 1, "tag": "classic", "timestamp": 123456}
    response = client.post("/tags", json=payload)
    assert response.status_code == status.HTTP_201_CREATED
    created = response.json()
    assert created["tag"] == payload["tag"]
    assert created["userId"] == payload["userId"]


def test_update_tag_changes_values(client: TestClient):
    payload = {"userId": 2, "movieId": 2, "tag": "updated tag", "timestamp": 222222}
    response = client.put("/tags/2", json=payload)
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["tag"] == payload["tag"]


def test_delete_tag_removes_record(client: TestClient):
    response = client.delete("/tags/1")
    assert response.status_code == status.HTTP_204_NO_CONTENT
    verify_response = client.get("/tags/1")
    assert verify_response.status_code == status.HTTP_404_NOT_FOUND
