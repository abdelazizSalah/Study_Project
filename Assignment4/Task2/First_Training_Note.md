## Training GAN Results:
- ![alt text](image.png)
- ![alt text](image-1.png)
- After training on the dataset, I found that D is saturated and almost predict all normal packets correctly, which G gets pushed farther instead of closer, so L2 distance between feature vectors keep increasing until it reaches stable ceilts almost 390. 
- So I will try now to fix this issue by modifing the learning rates of D and G.
- First inference results:
  - Testing on test data...
[D-mode] Precision: 0.9900 | Recall: 0.4193 | F1: 0.5891
- Second inference results with sigmoid at the end:
  -   [*] Phase 6 - Mode 1: Discriminator-based
  Unique labels: [0 1]
  [*] Number of normal validation samples: 13060
  [*] Collected 13060 validation scores
  [✓] D threshold: 1.000000
  Val scores stats | min=1.000000, mean=1.000000, max=1.000000
  Testing on test data...