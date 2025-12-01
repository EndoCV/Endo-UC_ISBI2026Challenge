
## Multi-modal Ulcerative Colitis Grading in Endoscopy - IEEE ISBI 2026 Challenge

Visit our [Website](https://endouc-cv.github.io)

**Challenge** on IEEE ISBI page [here](https://biomedicalimaging.org/2026/challenges/)

**Dataset** is available at #synapse [here](https://www.synapse.org/Synapse:syn70967760) 

#### Challenge tasks:

**Task I: Multi-class Mayo Scoring classification** accuracy (0-3) of the encoder architecture

**Task II: Image captioning** similarity compared to ground truth expert annotations on various severity present in the images for the decoder architecture.

**Task III: Generalisation of task 1 and task 2 on unseen centre** ulcerative colitis patient data.

### Evaluation metrics explained!
`` Please refer the file "evaluationMetrics.py" ``

#### Evaluation of your classification model

``The script calculates and prints two types of results: overall multiclass metrics and per-class metrics derived from the Confusion Matrix.``

``Options: 1 -> MES only; 2 -> UCEIS only; 3 -> MES and caption metrics``

1. Overall Multiclass Metrics: These metrics provide a single summary value for the entire classification task: Top-1 Accuracy, F1 Score (Macro) and AUC (Area Under the ROC Curve - Macro)

2.  Per-Class Metrics (Derived from Confusion Matrix): Precision per class, Recall per class and F1-Score per class

#### Evaluating your captions/descriptions from your model

``Your directory must contain two CSV files  structured as follows:``

- Ground Truth File: Include Two columns: ID with image name and ground-truth for captions description with header "captions" 

- Predictions File: Include Two columns: ID with image name and Predicted for caption description with header "Predictions"

**Metrics used**

1.	Cosine Similarity (TF-IDF)

2.	BLEU Score 

 - A classic n-gram precision metric.

Note: The code calculates unigram, bigram , trigram and standard BLEU. 

3.	METEOR Score: Considers exact matches, word stems, synonyms, and word order.


4.	ROUGE Scores: Measures word overlap.

- ROUGE-1: Measures unigram (single-word) overlap.

- ROUGE-L: Measures the Longest Common Subsequence (LCS), which captures sentence-level structure similarity.
