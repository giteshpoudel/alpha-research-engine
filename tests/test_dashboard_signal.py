import pytest
from fastapi.testclient import TestClient

from src.dashboard.app import create_app


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


def test_signal_page_and_api(client):
    resp = client.get("/signal")
    assert resp.status_code == 200
    assert 'href="/signal"' in resp.text  # nav link enabled

    api = client.get("/api/signal").json()
    assert isinstance(api, list)
