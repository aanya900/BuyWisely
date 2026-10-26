from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field, model_validator


router = APIRouter(
    prefix="/api/v1",
    tags=["Information Warnings"],
)


MISSING_VALUES = {
    "",
    "not provided",
    "not available",
    "not found",
    "unknown",
    "n/a",
    "na",
}


def normalize(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def is_missing(value: str | None) -> bool:
    return value is None or normalize(value) in MISSING_VALUES


class WarningRequest(BaseModel):
    product_title: str = Field(min_length=1, max_length=300)

    warranty: str | None = Field(default=None, max_length=1000)

    specifications: dict[str, str | None] = Field(
        default_factory=dict
    )

    # Explicitly supplied because important specifications
    # differ by product category.
    required_specifications: list[str] = Field(
        default_factory=list,
        max_length=30,
    )

    review_count: int | None = Field(default=None, ge=0)

    # This endpoint assumes a five-point rating scale.
    rating: float | None = Field(
        default=None,
        ge=0,
        le=5,
        allow_inf_nan=False,
    )

    # A configurable project heuristic, not a fraud threshold.
    low_review_threshold: int = Field(default=20, ge=1, le=10000)

    @model_validator(mode="after")
    def check_specification_names(self):
        names = [
            normalize(name)
            for name in self.specifications
        ]

        if any(not name for name in names):
            raise ValueError(
                "Specification names must not be blank."
            )

        if len(names) != len(set(names)):
            raise ValueError(
                "Specification names must be unique after normalization."
            )

        if any(
            not normalize(name)
            for name in self.required_specifications
        ):
            raise ValueError(
                "Required specification names must not be blank."
            )

        return self


class InformationWarning(BaseModel):
    code: str
    level: Literal["info", "caution"]
    message: str
    evidence: str


class WarningResponse(BaseModel):
    product_title: str
    warning_count: int
    warnings: list[InformationWarning]
    missing_required_specifications: list[str]
    assessment: str
    limitations: list[str]


@router.post("/warnings", response_model=WarningResponse)
def analyze_warnings(request: WarningRequest):
    warnings = []

    if is_missing(request.warranty):
        warnings.append(InformationWarning(
            code="WARRANTY_NOT_PROVIDED",
            level="caution",
            message=(
                "Warranty details are missing. "
                "Check the manufacturer or seller policy."
            ),
            evidence="The supplied warranty field is missing or uninformative.",
        ))

    normalized_specs = {
        normalize(name): value
        for name, value in request.specifications.items()
    }

    required_names = sorted({
        normalize(name)
        for name in request.required_specifications
    })

    missing_specs = [
        name
        for name in required_names
        if is_missing(normalized_specs.get(name))
    ]

    if missing_specs:
        warnings.append(InformationWarning(
            code="REQUIRED_SPECIFICATIONS_MISSING",
            level="caution",
            message=(
                "Some required specifications are not confirmed."
            ),
            evidence=", ".join(missing_specs),
        ))

    if not required_names:
        warnings.append(InformationWarning(
            code="SPECIFICATION_CHECK_NOT_CONFIGURED",
            level="info",
            message=(
                "No required specifications were supplied, "
                "so specification completeness was not assessed."
            ),
            evidence="required_specifications is empty.",
        ))

    if request.review_count is None:
        warnings.append(InformationWarning(
            code="REVIEW_COUNT_NOT_PROVIDED",
            level="info",
            message="The amount of review evidence is unknown.",
            evidence="review_count was not supplied.",
        ))

    elif request.review_count < request.low_review_threshold:
        warnings.append(InformationWarning(
            code="LIMITED_REVIEW_EVIDENCE",
            level="caution",
            message=(
                "The supplied review count is below the configured "
                "threshold. Treat rating-based conclusions cautiously."
            ),
            evidence=(
                f"Review count: {request.review_count}; "
                f"configured threshold: {request.low_review_threshold}."
            ),
        ))

    if request.rating is None:
        warnings.append(InformationWarning(
            code="RATING_NOT_PROVIDED",
            level="info",
            message="No overall rating was supplied.",
            evidence="rating was not supplied.",
        ))

    elif request.rating == 5:
        warnings.append(InformationWarning(
            code="PERFECT_RATING_CONTEXT",
            level="info",
            message=(
                "A perfect supplied rating does not establish quality "
                "or authenticity. Inspect the underlying reviews."
            ),
            evidence="The supplied rating is 5 out of 5.",
        ))

    if request.review_count == 0 and request.rating is not None:
        warnings.append(InformationWarning(
            code="RATING_COUNT_NEEDS_VERIFICATION",
            level="info",
            message=(
                "A rating was supplied with zero reviews. Verify whether "
                "the website distinguishes ratings from written reviews."
            ),
            evidence=(
                f"Rating: {request.rating}; review count: 0."
            ),
        ))

    return WarningResponse(
        product_title=request.product_title,
        warning_count=len(warnings),
        warnings=warnings,
        missing_required_specifications=missing_specs,
        assessment=(
            "Review the information gaps and context notes."
            if warnings
            else "No warnings were triggered by these configured checks."
        ),
        limitations=[
            "Only supplied data was checked.",
            "No product, seller or review authenticity was verified.",
            "The review-count threshold is a configurable heuristic.",
            "No warnings does not mean the product is safe or suitable.",
            "Price anomalies are not assessed by this endpoint.",
        ],
    )