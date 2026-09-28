| A                                   | B                       |   ndcg@10 A - B | 95% CI           |   p-value | significant (p < 0.05)   |
|:------------------------------------|:------------------------|----------------:|:-----------------|----------:|:-------------------------|
| BM25                                | TF-IDF                  |           0.026 | [+0.002, +0.050] |     0.045 | yes                      |
| Dense (BGE-small)                   | BM25                    |           0.023 | [-0.009, +0.056] |     0.162 | no                       |
| Dense (BGE-small)                   | Dense (MiniLM-L6)       |           0.061 | [+0.035, +0.088] |     0.000 | yes                      |
| Hybrid RRF (BM25 + BGE)             | Dense (BGE-small)       |           0.018 | [-0.006, +0.040] |     0.126 | no                       |
| Hybrid weighted, α=0.6 (BM25 + BGE) | Dense (BGE-small)       |           0.023 | [+0.005, +0.042] |     0.013 | yes                      |
| Hybrid weighted, α=0.6 (BM25 + BGE) | Hybrid RRF (BM25 + BGE) |           0.006 | [-0.010, +0.022] |     0.498 | no                       |
