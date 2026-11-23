| Model | Final train loss | Train time (s) | 5-gram copy rate | 3-gram repetition rate | Coherence / grammar |
|---|---|---|---|---|---|
| RNN | 0.960 | 48.0 | 27.1% | 5.5% | Short phrases are grammatical and some lines are joke shaped, but setups and punchlines do not match. Copies the most, and repeats one training phrase twice ("who was drying a glass"). |
| LSTM | 1.069 | 47.6 | 6.0% | 4.0% | Weakest grammar: many broken quote fragments (the" says" the street!) and questions with no real answer. Copies little because it fits the data least well (highest loss). |
| GRU | 0.724 | 59.8 | 18.1% | 5.5% | Best joke structure. Opens with a complete real joke ("what did the 0 say to the 8? nice belt!") and keeps the question and answer form. Clauses are mostly grammatical, but later lines drift into noise. |
| GRU 2-layer | 0.959 | 86.0 | 12.1% | 9.0% | Fluent first person sentences, but many repeat "i was ..." and lose meaning halfway through. The first joke is copied verbatim. Undertrained at 30 epochs. |

Shared settings: word-level, embedding 64, hidden 128, 30 epochs, batch 32, Adam lr 0.003, grad clip 5, seed 1, device cpu. Prompt "what did the", 200 tokens, temperature 0.8.

Copy rate: share of the story's 5-grams (prompt included) that appear verbatim in the training text. Repetition rate: share of the story's 3-grams that repeat an earlier 3-gram in the same story. Coherence notes are written by hand for this seed; rerunning the script resets them to TODO.
