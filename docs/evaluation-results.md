# BuyWisely Evaluation Results

## Backend Feature Tests

Command:
`python -m pytest tests/test_features.py -v`

Actual result:
<!-- Paste the actual summary here. -->

Scope:
- Product comparison
- Keyword-based review analysis
- Catalog-based recommendations
- Information warnings
- Request validation

Not covered:
- Browser extraction
- Main application startup and integration
- RAG retrieval quality
- Custom language model answer quality
- End-to-end response time

## Analyze Endpoint: Controlled Dependency Tests

Command:
`python -m pytest tests/test_analyze.py -v`

Actual result:
<!-- Paste the real pytest summary here. -->

Checked:
- Successful response and prompt construction
- Empty retrieved context
- Retrieval failure
- Model connection failure
- Model timeout
- Upstream HTTP errors
- Invalid JSON
- Empty generated answer
- Empty-question validation

Limitations:
- Retrieval and model responses were simulated.
- These tests do not measure model accuracy or hallucinations.
- These tests do not verify actual RAG startup or model-server connectivity.
- Browser extraction and extension behavior were not tested.