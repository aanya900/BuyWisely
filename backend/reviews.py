import re
from collections import Counter

from fastapi import APIRouter
from pydantic import BaseModel, Field


router = APIRouter(
    prefix="/api/v1/reviews",
    tags=["Review Analysis"],
)


class ReviewInput(BaseModel):
    review_id: str = Field(min_length=1, max_length=100)
    text: str = Field(min_length=1, max_length=5000)


class ReviewAnalysisRequest(BaseModel):
    product_title: str = Field(min_length=1, max_length=300)
    reviews: list[ReviewInput] = Field(
        min_length=1,
        max_length=100,
    )


# Small, explicit phrase lists for the baseline.
# These are not a trained sentiment model.
POSITIVE_PHRASES = (
    "good battery",
    "great battery",
    "comfortable",
    "clear sound",
    "good sound",
    "easy to use",
    "works well",
    "good value",
)

NEGATIVE_PHRASES = (
    "poor battery",
    "uncomfortable",
    "connection drops",
    "disconnects",
    "stopped working",
    "poor sound",
    "bad sound",
    "poor microphone",
    "too expensive",
)


def contains_phrase(text: str, phrase: str) -> bool:
    pattern = r"\b" + re.escape(phrase) + r"\b"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


@router.post("/analyze")
def analyze_reviews(request: ReviewAnalysisRequest):
    positive_excerpts = []
    negative_excerpts = []
    complaint_counts = Counter()

    counts = {
        "positive": 0,
        "negative": 0,
        "mixed": 0,
        "unclassified": 0,
    }

    seen_texts = set()
    duplicate_count = 0

    for review in request.reviews:
        normalized = " ".join(review.text.casefold().split())

        # Avoid counting identical submitted review text twice.
        if normalized in seen_texts:
            duplicate_count += 1
            continue

        seen_texts.add(normalized)

        positive_matches = [
            phrase
            for phrase in POSITIVE_PHRASES
            if contains_phrase(review.text, phrase)
        ]

        negative_matches = [
            phrase
            for phrase in NEGATIVE_PHRASES
            if contains_phrase(review.text, phrase)
        ]

        if positive_matches and negative_matches:
            label = "mixed"
        elif positive_matches:
            label = "positive"
        elif negative_matches:
            label = "negative"
        else:
            label = "unclassified"

        counts[label] += 1

        if positive_matches:
            positive_excerpts.append({
                "review_id": review.review_id,
                "text": review.text,
                "matched_phrases": positive_matches,
            })

        if negative_matches:
            negative_excerpts.append({
                "review_id": review.review_id,
                "text": review.text,
                "matched_phrases": negative_matches,
            })

            # Each phrase is counted at most once per review.
            complaint_counts.update(negative_matches)

    classified_count = (
        counts["positive"]
        + counts["negative"]
        + counts["mixed"]
    )

    if classified_count == 0:
        overall_sentiment = "insufficient_evidence"
    elif (
        counts["mixed"] > 0
        or (counts["positive"] > 0 and counts["negative"] > 0)
    ):
        overall_sentiment = "mixed"
    elif counts["positive"] > 0:
        overall_sentiment = "positive"
    else:
        overall_sentiment = "negative"

    recurring_complaints = [
        {
            "phrase": phrase,
            "review_count": count,
        }
        for phrase, count in complaint_counts.most_common()
        if count >= 2
    ]

    return {
        "product_title": request.product_title,
        "method": "keyword_baseline",
        "submitted_review_count": len(request.reviews),
        "analyzed_review_count": len(seen_texts),
        "duplicate_texts_skipped": duplicate_count,
        "sentiment_counts": counts,
        "overall_sentiment": overall_sentiment,
        "positive_excerpts": positive_excerpts,
        "negative_excerpts": negative_excerpts,
        "recurring_complaint_phrases": recurring_complaints,
        "limitations": [
            "Results describe only the supplied reviews.",
            "Keyword rules can misinterpret negation and sarcasm.",
            "Unmatched reviews are unclassified, not neutral.",
            "Complaint phrases are not grouped into semantic themes.",
            "This endpoint does not detect fake reviews.",
            "This is not an LLM-generated review summary.",
        ],
    }