# Conditional Mammography Synthesis & Classification  
## (EMBED + Class-Conditional DDPM)

This repository reproduces the experiments from:

> **Utility of AI-Generated Images in Disease Classification: A Study in Breast Cancer Mammography**

It implements a full pipeline to:

- Construct a binary EMBED dataset (benign vs malignant)
- Train a class-conditional DDPM
- Generate synthetic mammograms (classifier-free guidance)
- Train an Oracle classifier (ConvNeXt-Tiny)
- Perform synthetic augmentation experiments
- Run two-phase training (real + synthetic → real fine-tune)
- Evaluate Balanced Accuracy, Sensitivity, Specificity, and FID
