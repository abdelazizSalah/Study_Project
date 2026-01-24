## Final Takeaway from the Results 

The results show a **clear, scenario-dependent impact** of incorporating statistical features (`use_stats = 1`) into the model.

### Scenario 2 (S2: Partial-attack training)
- **Recall:** Using raw sequence features only (`use_stats = 0`) consistently achieves **higher recall**, especially for `raw`, `re15`, and `re5`. This indicates that the model without stats is better at **detecting attacks** (fewer false negatives).
- **Precision:** Adding stats (`use_stats = 1`) generally **improves precision**, particularly for `re10` and `re5`, meaning fewer false positives.
- **F1-score:** The trade-off becomes clear in F1:
  - `use_stats = 1` improves F1 for `re10` and `re5`.
  - `use_stats = 0` remains better for `raw` and `re15`.

**Interpretation (S2):**  
Statistical features help the model become more conservative (higher precision), but at the cost of missing more attacks (lower recall). Whether stats are beneficial depends on whether **precision or recall is the primary objective**.

---

### Scenario 3 (S3: Unseen-attack training)
- **Recall:** Overall recall is **very low across all representations**, indicating the difficulty of Scenario 3. Stats provide **slight recall gains** for `re10`, but performance remains weak overall.
- **Precision:** Statistical features **improve precision** across `re10`, `re15`, and `re5`, while models without stats are near-zero.
- **F1-score:** Only configurations with `use_stats = 1` achieve **non-trivial F1-scores**. Without stats, the model is largely ineffective.

**Interpretation (S3):**  
In the hardest generalization setting, **statistical features are essential**. While absolute performance remains low, stats are the difference between *near-random behavior* and *meaningful detection*.

---

### Overall Conclusion
- **Without stats (`use_stats = 0`)**:
  - Better recall in easier settings (Scenario 2)
  - Poor robustness in hard generalization (Scenario 3)
- **With stats (`use_stats = 1`)**:
  - Improves precision and stability
  - Enables non-zero performance in Scenario 3
  - Introduces a recall–precision trade-off in Scenario 2

**Final takeaway:**  
Statistical features act as a **regularizing signal**. They reduce over-sensitivity to raw byte patterns, improving robustness and precision—especially for unseen attacks—but may suppress recall when attacks are already well represented in training data.
