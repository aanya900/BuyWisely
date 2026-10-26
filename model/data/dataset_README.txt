BuyWisely synthetic starter dataset

1,440 examples from 240 fictional product profiles, with six questions per profile.
Categories: wireless earbuds, laptops, smartphones, power banks.
Prices, ratings and specifications are fictional. This is not scraped customer data.
Train: 1,152 examples (192 products).
Validation: 144 examples (24 products).
Test: 144 examples (24 products).
Splits are grouped by product, not by individual question. Answer patterns are shared across splits; this is not evidence of general reasoning or robustness.

Use the training text as model/data/shopping_data.txt for an initial pipeline test.
Do not concatenate validation or test records into training data.
Preserve validation and test data for evaluation. Check individual examples before use.

Limitations: template-generated, low linguistic diversity, four categories only, no real reviews, no review summaries, no comparison-pair training, no live pricing, no production-quality guarantee.
This dataset tests basic grounded QA and missing-information behavior. It is not enough to train a reliable shopping assistant from scratch.
The earlier 128-character context setting is too short for these records. Measure tokenized record lengths and choose context length and model size based on your hardware. Train and inference must use matching model configurations.
For character-level tokenization, include an unknown-character token rather than silently dropping unseen characters. Save vocabulary as JSON and model weights as a state_dict, rather than pickling the tokenizer object.
