from fastapi import APIRouter
from pydantic import BaseModel, Field


router = APIRouter(
    prefix="/api/v1",
    tags=["Product Comparison"],
)


class Product(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    category: str = Field(min_length=1, max_length=100)

    # Use a number, not a formatted string such as "₹2,499".
    price: float | None = Field(
        default=None,
        gt=0,
        allow_inf_nan=False,
    )

    currency: str = Field(
        default="INR",
        min_length=3,
        max_length=3,
    )

    specifications: dict[str, str] = Field(default_factory=dict)


class CompareRequest(BaseModel):
    product_a: Product
    product_b: Product


@router.post("/compare")
def compare_products(request: CompareRequest):
    a = request.product_a
    b = request.product_b

    warnings = []
    price_comparison = {
        "status": "unavailable",
        "lower_price_product": None,
        "difference": None,
        "currency": None,
        "explanation": "",
    }

    # Compare prices only when both are present and currencies match.
    if a.price is None or b.price is None:
        price_comparison["explanation"] = (
            "Both product prices are required for a price comparison."
        )

    elif a.currency.upper() != b.currency.upper():
        price_comparison["explanation"] = (
            "The prices use different currencies. "
            "No currency conversion has been performed."
        )

    else:
        price_comparison["status"] = "available"
        price_comparison["currency"] = a.currency.upper()
        price_comparison["difference"] = round(
            abs(a.price - b.price),
            2,
        )

        if a.price < b.price:
            price_comparison["lower_price_product"] = "product_a"
            price_comparison["explanation"] = (
                f"{a.title} has the lower listed price."
            )
        elif b.price < a.price:
            price_comparison["lower_price_product"] = "product_b"
            price_comparison["explanation"] = (
                f"{b.title} has the lower listed price."
            )
        else:
            price_comparison["explanation"] = (
                "Both products have the same listed price."
            )

    # Show supplied specifications side by side.
    specification_names = sorted(
        set(a.specifications) | set(b.specifications)
    )

    specification_comparison = []

    for name in specification_names:
        value_a = a.specifications.get(name)
        value_b = b.specifications.get(name)

        specification_comparison.append({
            "specification": name,
            "product_a": value_a or "Not provided",
            "product_b": value_b or "Not provided",
        })

    if a.category.strip().casefold() != b.category.strip().casefold():
        warnings.append(
            "The supplied categories differ. "
            "These products may not be directly comparable."
        )

    if not specification_names:
        warnings.append(
            "No specifications were supplied for either product."
        )
    elif any(
        not a.specifications.get(name)
        or not b.specifications.get(name)
        for name in specification_names
    ):
        warnings.append(
            "Some specifications are missing for one or both products."
        )

    warnings.append(
        "A lower price does not establish better quality or value. "
        "Delivery charges, seller reliability and current availability "
        "have not been verified."
    )

    return {
        "product_a": a.model_dump(),
        "product_b": b.model_dump(),
        "price_comparison": price_comparison,
        "specification_comparison": specification_comparison,
        "warnings": warnings,
        "evidence_basis": "User-supplied product data only",
    }