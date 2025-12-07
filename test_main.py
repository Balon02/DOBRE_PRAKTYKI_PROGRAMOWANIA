import os

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

import main


def seed_database(session: Session) -> None:
    admin_user = main.User(
        username="admin",
        hashed_password=main.hash_password("admin123"),
        roles=main._serialize_roles([main.ADMIN_ROLE]),
    )
    regular_user = main.User(
        username="user",
        hashed_password=main.hash_password("user123"),
        roles=main._serialize_roles([main.DEFAULT_USER_ROLE]),
    )
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
    session.add_all([admin_user, regular_user] + movies + links + ratings + tags)
    session.commit()


@pytest.fixture()
def client(tmp_path):
    db_path = tmp_path / "test.sqlite"
    os.environ["SKIP_DB_POPULATE"] = "1"
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


def login_and_get_token(client: TestClient, username: str, password: str) -> str:
    response = client.post("/login", json={"username": username, "password": password})
    assert response.status_code == status.HTTP_200_OK
    return response.json()["access_token"]


@pytest.fixture()
def admin_headers(client: TestClient):
    token = login_and_get_token(client, "admin", "admin123")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def user_headers(client: TestClient):
    token = login_and_get_token(client, "user", "user123")
    return {"Authorization": f"Bearer {token}"}


def test_get_movies_returns_all(client: TestClient, admin_headers):
    response = client.get("/movies", headers=admin_headers)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert len(data) == 2
    assert data[0]["title"] == "Toy Story (1995)"


def test_get_movie_by_id_returns_item(client: TestClient, admin_headers):
    response = client.get("/movies/1", headers=admin_headers)
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["movieId"] == 1


def test_get_movie_by_id_missing_returns_404(client: TestClient, admin_headers):
    response = client.get("/movies/999", headers=admin_headers)
    assert response.status_code == status.HTTP_404_NOT_FOUND


def test_create_movie_adds_record(client: TestClient, admin_headers):
    payload = {"movieId": 3, "title": "Apollo 13 (1995)", "genres": "Drama"}
    response = client.post("/movies", json=payload, headers=admin_headers)
    assert response.status_code == status.HTTP_201_CREATED
    created = response.json()
    assert created["movieId"] == payload["movieId"]
    assert created["title"] == payload["title"]

    list_response = client.get("/movies", headers=admin_headers)
    assert len(list_response.json()) == 3


def test_update_movie_persists_changes(client: TestClient, admin_headers):
    payload = {"title": "Toy Story (Updated)", "genres": "Adventure"}
    response = client.put("/movies/1", json=payload, headers=admin_headers)
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["title"] == payload["title"]

    verify_response = client.get("/movies/1", headers=admin_headers)
    assert verify_response.json()["genres"] == payload["genres"]


def test_delete_movie_removes_record(client: TestClient, admin_headers):
    response = client.delete("/movies/2", headers=admin_headers)
    assert response.status_code == status.HTTP_204_NO_CONTENT

    missing_response = client.get("/movies/2", headers=admin_headers)
    assert missing_response.status_code == status.HTTP_404_NOT_FOUND


def test_get_link_by_id_returns_item(client: TestClient, admin_headers):
    response = client.get("/links/1", headers=admin_headers)
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["tmdbId"] == "862"


def test_get_link_missing_returns_404(client: TestClient, admin_headers):
    response = client.get("/links/999", headers=admin_headers)
    assert response.status_code == status.HTTP_404_NOT_FOUND


def test_create_link_adds_record(client: TestClient, admin_headers):
    payload = {"movieId": 3, "imdbId": "94943", "tmdbId": "9480"}
    response = client.post("/links", json=payload, headers=admin_headers)
    assert response.status_code == status.HTTP_201_CREATED
    assert response.json()["movieId"] == payload["movieId"]

    verify_response = client.get("/links/3", headers=admin_headers)
    assert verify_response.status_code == status.HTTP_200_OK


def test_update_link_persists_changes(client: TestClient, admin_headers):
    payload = {"imdbId": "114709-ALT", "tmdbId": "862"}
    response = client.put("/links/1", json=payload, headers=admin_headers)
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["imdbId"] == payload["imdbId"]


def test_delete_link_removes_record(client: TestClient, admin_headers):
    response = client.delete("/links/2", headers=admin_headers)
    assert response.status_code == status.HTTP_204_NO_CONTENT
    verify_response = client.get("/links/2", headers=admin_headers)
    assert verify_response.status_code == status.HTTP_404_NOT_FOUND


def test_get_ratings_returns_all(client: TestClient, admin_headers):
    response = client.get("/ratings", headers=admin_headers)
    assert response.status_code == status.HTTP_200_OK
    assert len(response.json()) == 2


def test_get_rating_by_id_returns_item(client: TestClient, admin_headers):
    response = client.get("/ratings/1", headers=admin_headers)
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["rating"] == 4.5


def test_get_rating_missing_returns_404(client: TestClient, admin_headers):
    response = client.get("/ratings/999", headers=admin_headers)
    assert response.status_code == status.HTTP_404_NOT_FOUND


def test_create_rating_adds_record(client: TestClient, admin_headers):
    payload = {"userId": 3, "movieId": 1, "rating": 5.0, "timestamp": 111111}
    response = client.post("/ratings", json=payload, headers=admin_headers)
    assert response.status_code == status.HTTP_201_CREATED
    created = response.json()
    assert created["userId"] == payload["userId"]
    assert created["rating"] == payload["rating"]


def test_update_rating_changes_values(client: TestClient, admin_headers):
    payload = {"userId": 1, "movieId": 1, "rating": 2.5, "timestamp": 222222}
    response = client.put("/ratings/1", json=payload, headers=admin_headers)
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["rating"] == payload["rating"]

    verify_response = client.get("/ratings/1", headers=admin_headers)
    assert verify_response.json()["timestamp"] == payload["timestamp"]


def test_delete_rating_removes_record(client: TestClient, admin_headers):
    response = client.delete("/ratings/2", headers=admin_headers)
    assert response.status_code == status.HTTP_204_NO_CONTENT
    verify_response = client.get("/ratings/2", headers=admin_headers)
    assert verify_response.status_code == status.HTTP_404_NOT_FOUND


def test_get_tags_returns_all(client: TestClient, admin_headers):
    response = client.get("/tags", headers=admin_headers)
    assert response.status_code == status.HTTP_200_OK
    assert len(response.json()) == 2


def test_get_tag_by_id_returns_item(client: TestClient, admin_headers):
    response = client.get("/tags/1", headers=admin_headers)
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["tag"] == "funny"


def test_get_tag_missing_returns_404(client: TestClient, admin_headers):
    response = client.get("/tags/999", headers=admin_headers)
    assert response.status_code == status.HTTP_404_NOT_FOUND


def test_create_tag_adds_record(client: TestClient, admin_headers):
    payload = {"userId": 3, "movieId": 1, "tag": "classic", "timestamp": 123456}
    response = client.post("/tags", json=payload, headers=admin_headers)
    assert response.status_code == status.HTTP_201_CREATED
    created = response.json()
    assert created["tag"] == payload["tag"]
    assert created["userId"] == payload["userId"]


def test_update_tag_changes_values(client: TestClient, admin_headers):
    payload = {"userId": 2, "movieId": 2, "tag": "updated tag", "timestamp": 222222}
    response = client.put("/tags/2", json=payload, headers=admin_headers)
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["tag"] == payload["tag"]


def test_delete_tag_removes_record(client: TestClient, admin_headers):
    response = client.delete("/tags/1", headers=admin_headers)
    assert response.status_code == status.HTTP_204_NO_CONTENT
    verify_response = client.get("/tags/1", headers=admin_headers)
    assert verify_response.status_code == status.HTTP_404_NOT_FOUND


def test_login_success_returns_token(client: TestClient):
    response = client.post("/login", json={"username": "admin", "password": "admin123"})
    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]


def test_login_invalid_credentials(client: TestClient):
    response = client.post("/login", json={"username": "admin", "password": "wrong"})
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_create_user_requires_admin(client: TestClient, user_headers):
    payload = {"username": "newbie", "password": "newpass", "roles": ["ROLE_USER"]}
    response = client.post("/users", json=payload, headers=user_headers)
    assert response.status_code == status.HTTP_403_FORBIDDEN


def test_create_user_with_admin_allows_creation(client: TestClient, admin_headers):
    payload = {"username": "another", "password": "anotherpass", "roles": ["ROLE_USER"]}
    response = client.post("/users", json=payload, headers=admin_headers)
    assert response.status_code == status.HTTP_201_CREATED
    created = response.json()
    assert created["username"] == payload["username"]

    login_response = client.post(
        "/login", json={"username": payload["username"], "password": payload["password"]}
    )
    assert login_response.status_code == status.HTTP_200_OK


def test_user_details_requires_token(client: TestClient):
    response = client.get("/user_details")
    assert response.status_code in {status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN}


def test_user_details_returns_payload(client: TestClient, user_headers):
    response = client.get("/user_details", headers=user_headers)
    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert body["username"] == "user"
    assert "ROLE_USER" in body["roles"]
