from copy import deepcopy

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from comparison import router as comparison_router
from reviews import router as reviews_router
from recommendations import router as recommendations_router
from warnings_analysis import router as warnings_router


@pytest.fixture
def client():
    app = FastAPI()

    app.include_router(comparison_router)
    app.include_router(reviews_router)
    app.include_router(recommendations_router)
    app.include_router(warnings_router)

    with TestClient(app) as test_client:
        yield test_client


def comparison_payload():
    return {
        "product_a": {
            "title": "Example Earbuds A",
            "category": "wireless_earbuds",
            "price": 2499,
            "currency": "INR",
            "specifications": {
                "battery_life": "30 hours including case",
                "water_resistance": "Not provided",
            },
        },
        "product_b": {
            "title": "Example Earbuds B",
            "category": "wireless_earbuds",
            "price": 2999,
            "currency": "INR",
            "specifications": {
                "battery_life": "40 hours including case",
                "water_resistance": "IPX4 as listed by seller",
            },
        },
    }


def recommendation_payload():
    return {
        "budget": 3000,
        "currency": "INR",
        "category": "wireless_earbuds",
        "required_features": ["usb-c"],
        "limit": 5,
        "products": [
            {
                "product_id": "a",
                "title": "Example A",
                "category": "wireless_earbuds",
                "price": 2499,
                "currency": "INR",
                "features": ["usb-c", "wireless"],
            },
            {
                "product_id": "b",
                "title": "Example B",
                "category": "wireless_earbuds",
                "price": 2999,
                "currency": "INR",
                "features": [
                    "usb-c",
                    "wireless",
                    "active noise cancellation",
                ],
            },
            {
                "product_id": "c",
                "title": "Example C",
                "category": "wireless_earbuds",
                "price": 3999,
                "currency": "INR",
                "features": ["usb-c"],
            },
            {
                "product_id": "d",
                "title": "Example D",
                "category": "wireless_earbuds",
                "price": 1799,
                "currency": "INR",
                "features": ["wireless"],
            },
        ],
    }


def complete_warning_payload():
    return {
        "product_title": "Example Earbuds",
        "warranty": "12 months as listed by seller",
        "specifications": {
            "battery_life": "30 hours including case",
            "water_resistance": "IPX4 as listed by seller",
        },
        "required_specifications": [
            "battery_life",
            "water_resistance",
        ],
        "review_count": 150,
        "rating": 4.2,
        "low_review_threshold": 20,
    }


# -----------------------
# Comparison tests
# -----------------------

def test_compare_price_difference(client):
    response = client.post(
        "/api/v1/compare",
        json=comparison_payload(),
    )

    assert response.status_code == 200
    result = response.json()["price_comparison"]

    assert result["lower_price_product"] == "product_a"
    assert result["difference"] == 500
    assert result["currency"] == "INR"


def test_compare_missing_price(client):
    payload = comparison_payload()
    payload["product_a"]["price"] = None

    response = client.post("/api/v1/compare", json=payload)

    assert response.status_code == 200
    result = response.json()["price_comparison"]

    assert result["status"] == "unavailable"
    assert result["lower_price_product"] is None


def test_compare_currency_mismatch(client):
    payload = comparison_payload()
    payload["product_b"]["currency"] = "USD"

    response = client.post("/api/v1/compare", json=payload)

    assert response.status_code == 200
    assert response.json()["price_comparison"]["status"] == "unavailable"


def test_compare_missing_placeholder_warning(client):
    response = client.post(
        "/api/v1/compare",
        json=comparison_payload(),
    )

    assert response.status_code == 200
    assert (
        "Some specifications are missing for one or both products."
        in response.json()["warnings"]
    )


# -----------------------
# Review analysis tests
# -----------------------

def test_reviews_mixed_sentiment_and_repeated_phrase(client):
    payload = {
        "product_title": "Example Earbuds",
        "reviews": [
            {
                "review_id": "r1",
                "text": "Comfortable fit and clear sound.",
            },
            {
                "review_id": "r2",
                "text": "The connection drops during calls.",
            },
            {
                "review_id": "r3",
                "text": "Good battery, but the connection drops outdoors.",
            },
            {
                "review_id": "r4",
                "text": "The packaging arrived yesterday.",
            },
        ],
    }

    response = client.post(
        "/api/v1/reviews/analyze",
        json=payload,
    )

    assert response.status_code == 200
    data = response.json()

    assert data["overall_sentiment"] == "mixed"
    assert data["sentiment_counts"] == {
        "positive": 1,
        "negative": 1,
        "mixed": 1,
        "unclassified": 1,
    }
    assert data["recurring_complaint_phrases"] == [
        {"phrase": "connection drops", "review_count": 2}
    ]

    submitted_texts = {
        review["text"] for review in payload["reviews"]
    }

    for excerpt in (
        data["positive_excerpts"] + data["negative_excerpts"]
    ):
        assert excerpt["text"] in submitted_texts


def test_reviews_skip_duplicate_text(client):
    response = client.post(
        "/api/v1/reviews/analyze",
        json={
            "product_title": "Example Earbuds",
            "reviews": [
                {"review_id": "r1", "text": "Connection drops."},
                {"review_id": "r2", "text": "Connection drops."},
            ],
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert data["analyzed_review_count"] == 1
    assert data["duplicate_texts_skipped"] == 1
    assert data["recurring_complaint_phrases"] == []


def test_reviews_unclassified_is_not_neutral(client):
    response = client.post(
        "/api/v1/reviews/analyze",
        json={
            "product_title": "Example Product",
            "reviews": [
                {
                    "review_id": "r1",
                    "text": "The box contains a cable.",
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["overall_sentiment"] == "insufficient_evidence"


# -----------------------
# Recommendation tests
# -----------------------

def test_recommendations_filter_and_sort(client):
    response = client.post(
        "/api/v1/recommend",
        json=recommendation_payload(),
    )

    assert response.status_code == 200
    data = response.json()

    assert data["matching_product_count"] == 2
    assert [
        item["product_id"]
        for item in data["recommendations"]
    ] == ["a", "b"]


def test_recommendations_no_matches(client):
    payload = recommendation_payload()
    payload["budget"] = 1000

    response = client.post("/api/v1/recommend", json=payload)

    assert response.status_code == 200
    assert response.json()["status"] == "no_matches"
    assert response.json()["recommendations"] == []


def test_recommendations_include_exact_budget(client):
    payload = recommendation_payload()
    payload["budget"] = 2499

    response = client.post("/api/v1/recommend", json=payload)

    assert response.status_code == 200
    assert [
        item["product_id"]
        for item in response.json()["recommendations"]
    ] == ["a"]


def test_recommendations_reject_duplicate_ids(client):
    payload = recommendation_payload()
    duplicate = deepcopy(payload["products"][0])
    payload["products"].append(duplicate)

    response = client.post("/api/v1/recommend", json=payload)

    assert response.status_code == 422


# -----------------------
# Information warning tests
# -----------------------

def test_warnings_complete_information(client):
    response = client.post(
        "/api/v1/warnings",
        json=complete_warning_payload(),
    )

    assert response.status_code == 200
    assert response.json()["warning_count"] == 0


def test_warnings_missing_information(client):
    payload = complete_warning_payload()
    payload["warranty"] = "Not provided"
    payload["specifications"]["water_resistance"] = "Not provided"
    payload["review_count"] = 8
    payload["rating"] = 5

    response = client.post("/api/v1/warnings", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert data["warning_count"] == 4
    assert data["missing_required_specifications"] == [
        "water_resistance"
    ]
    assert {
        warning["code"] for warning in data["warnings"]
    } == {
        "WARRANTY_NOT_PROVIDED",
        "REQUIRED_SPECIFICATIONS_MISSING",
        "LIMITED_REVIEW_EVIDENCE",
        "PERFECT_RATING_CONTEXT",
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("rating", 6),
        ("review_count", -1),
    ],
)
def test_warnings_reject_invalid_values(client, field, value):
    payload = complete_warning_payload()
    payload[field] = value

    response = client.post("/api/v1/warnings", json=payload)

    assert response.status_code == 422