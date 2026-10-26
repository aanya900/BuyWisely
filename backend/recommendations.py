from fastapi import APIRouter
from pydantic import BaseModel, Field, model_validator


router = APIRouter(
    prefix="/api/v1",
    tags=["Recommendations"],
)


def normalize(value: str) -> str:
    return " ".join(value.strip().casefold().split())


class CatalogProduct(BaseModel):
    product_id: str = Field(min_length=1, max_length=100)
    title: str = Field(min_length=1, max_length=300)
    category: str = Field(min_length=1, max_length=100)

    price: float = Field(
        gt=0,
        allow_inf_nan=False,
    )

    currency: str = Field(
        default="INR",
        min_length=3,
        max_length=3,
    )

    # Features must be explicitly supplied.
    # Example: ["usb-c", "wireless", "active noise cancellation"]
    features: list[str] = Field(default_factory=list, max_length=100)


class RecommendationRequest(BaseModel):
    budget: float = Field(
        gt=0,
        allow_inf_nan=False,
    )

    currency: str = Field(
        default="INR",
        min_length=3,
        max_length=3,
    )

    category: str = Field(min_length=1, max_length=100)

    required_features: list[str] = Field(
        default_factory=list,
        max_length=20,
    )

    products: list[CatalogProduct] = Field(
        min_length=1,
        max_length=200,
    )

    limit: int = Field(default=5, ge=1, le=20)

    @model_validator(mode="after")
    def validate_catalog(self):
        product_ids = [product.product_id for product in self.products]

        if len(product_ids) != len(set(product_ids)):
            raise ValueError("Each product_id must be unique.")

        if not normalize(self.category):
            raise ValueError("Category must not be blank.")

        if any(not normalize(feature) for feature in self.required_features):
            raise ValueError("Required features must not be blank.")

        return self


@router.post("/recommend")
def recommend_products(request: RecommendationRequest):
    required = {
        normalize(feature)
        for feature in request.required_features
    }

    matches = []
    excluded = []

    for product in request.products:
        reasons = []

        if normalize(product.category) != normalize(request.category):
            reasons.append("Category does not match.")

        same_currency = (
            product.currency.upper() == request.currency.upper()
        )

        if not same_currency:
            reasons.append(
                "Currency does not match; no conversion was performed."
            )
        elif product.price > request.budget:
            reasons.append("Listed price exceeds the budget.")

        available_features = {
            normalize(feature)
            for feature in product.features
        }

        missing_features = sorted(required - available_features)

        if missing_features:
            reasons.append(
                "Required features are not confirmed in the supplied "
                "feature list: " + ", ".join(missing_features)
            )

        if reasons:
            excluded.append({
                "product_id": product.product_id,
                "title": product.title,
                "reasons": reasons,
            })
            continue

        matches.append({
            "product_id": product.product_id,
            "title": product.title,
            "price": product.price,
            "currency": product.currency.upper(),
            "features": product.features,
            "budget_remaining": round(
                request.budget - product.price,
                2,
            ),
            "reason": (
                "Matches the supplied category, is within budget, "
                "and includes all required feature labels."
            ),
        })

    # Lowest listed price first; use product ID to break ties.
    matches.sort(
        key=lambda product: (
            product["price"],
            product["product_id"],
        )
    )

    return {
        "status": "matches_found" if matches else "no_matches",
        "method": "catalog_filter",
        "sort_order": "price_ascending",
        "matching_product_count": len(matches),
        "recommendations": matches[:request.limit],
        "excluded_products": excluded,
        "limitations": [
            "Only the supplied catalog was searched.",
            "Feature matching uses normalized exact labels, not synonyms.",
            "Unlisted features are unconfirmed, not necessarily absent.",
            "Lowest price does not mean best quality or overall value.",
            "Prices, availability and product claims were not verified.",
        ],
    }