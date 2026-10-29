# BuyWisely Evaluation Results

All pytest commands in this report are run from the backend directory.

## Backend Feature Tests

Command:
`python -m pytest tests/test_features.py -v`

Actual result:
============================================================== test session starts ==============================================================
platform win32 -- Python 3.13.13, pytest-9.1.1, pluggy-1.7.0 -- D:\extension-project\BuyWisely\backend\venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: D:\extension-project\BuyWisely\backend
plugins: anyio-4.15.1
collected 15 items                                                                                                                               

tests/test_features.py::test_compare_price_difference PASSED                                                                               [  6%]
tests/test_features.py::test_compare_missing_price PASSED                                                                                  [ 13%]
tests/test_features.py::test_compare_currency_mismatch PASSED                                                                              [ 20%]
tests/test_features.py::test_compare_missing_placeholder_warning PASSED                                                                    [ 26%]
tests/test_features.py::test_reviews_mixed_sentiment_and_repeated_phrase PASSED                                                            [ 33%]
tests/test_features.py::test_reviews_skip_duplicate_text PASSED                                                                            [ 40%]
tests/test_features.py::test_reviews_unclassified_is_not_neutral PASSED                                                                    [ 46%]
tests/test_features.py::test_recommendations_filter_and_sort PASSED                                                                        [ 53%]
tests/test_features.py::test_recommendations_no_matches PASSED                                                                             [ 60%]
tests/test_features.py::test_recommendations_include_exact_budget PASSED                                                                   [ 66%]
tests/test_features.py::test_recommendations_reject_duplicate_ids PASSED                                                                   [ 73%]
tests/test_features.py::test_warnings_complete_information PASSED                                                                          [ 80%]
tests/test_features.py::test_warnings_missing_information PASSED                                                                           [ 86%]
tests/test_features.py::test_warnings_reject_invalid_values[rating-6] PASSED                                                               [ 93%]
tests/test_features.py::test_warnings_reject_invalid_values[review_count--1] PASSED                                                        [100%]

============================================================== 15 passed in 1.43s ===============================================================


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
============================================================== test session starts ==============================================================
platform win32 -- Python 3.13.13, pytest-9.1.1, pluggy-1.7.0 -- D:\extension-project\BuyWisely\backend\venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: D:\extension-project\BuyWisely\backend
plugins: anyio-4.15.1
collected 9 items                                                                                                                                

tests/test_analyze.py::test_analyze_success PASSED                                                                                         [ 11%]
tests/test_analyze.py::test_analyze_empty_context PASSED                                                                                   [ 22%]
tests/test_analyze.py::test_analyze_retrieval_failure PASSED                                                                               [ 33%]
tests/test_analyze.py::test_analyze_connection_errors[ConnectionError-503] PASSED                                                          [ 44%]
tests/test_analyze.py::test_analyze_connection_errors[Timeout-504] PASSED                                                                  [ 55%]
tests/test_analyze.py::test_analyze_model_http_error PASSED                                                                                [ 66%]
tests/test_analyze.py::test_analyze_invalid_json PASSED                                                                                    [ 77%]
tests/test_analyze.py::test_analyze_empty_answer PASSED                                                                                    [ 88%]
tests/test_analyze.py::test_analyze_blank_question PASSED                                                                                  [100%]

=============================================================== 9 passed in 2.20s ===============================================================


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


## Custom Model V2 Training Baseline

Tokenizer:
UTF-8 byte tokenizer, vocabulary size 256.

Model:
Decoder-only TinyGPT trained from random initialization.

Configuration:
- Embedding dimension: 64
- Attention heads: 4
- Transformer layers: 2
- Context length: 512 byte tokens

Training:
- Steps: 200
- Batch size: [actual value]
- Device: [actual device]
- Best validation loss: [actual result]

Limitations:
- Synthetic, template-based training data.
- Short context compared with full RAG prompts.
- No end-of-answer token.
- No demonstrated shopping-answer accuracy yet.
- Validation loss does not establish factual reliability.