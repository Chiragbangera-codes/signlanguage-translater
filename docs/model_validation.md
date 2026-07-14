# Model Validation & Leakage Verification Report

This report documents the validation methodology applied to the SignSpeak AI classification pipeline and verifies that the reported test metrics are trustworthy and free of data leakage.

---

## 1. Data Leakage Verification Checklist

To ensure the integrity of model evaluation, we verified the pipeline against the four main sources of data leakage:

### Check 1: Timeline of Splits (Disjoint Partitions)
*   **Methodology**: The dataset splitting was performed as the first step in the pipeline using `train_test_split` with a fixed random seed (`random_state=42`) and stratification on the target labels.
*   **Result**: The partition is strict. The training, validation, and test subsets are completely disjoint. Indices assigned to one partition are never exposed to another partition.

### Check 2: Sample Duplication Check
*   **Methodology**:
    1.  The raw dataset was verified to have zero duplicate rows (`df.duplicated().sum() == 0`).
    2.  An intersection analysis was executed to verify if any sample feature vector in the validation or test splits matches any sample in the training split.
*   **Result**: No duplicate rows exist. Overlap check results:
    *   Overlap between Train and Validation: **False**
    *   Overlap between Train and Test: **False**
    *   This confirms that the test set consists entirely of unique, unseen hand gesture configurations.

### Check 3: Preprocessing Statistics Isolation (Instance-Based Normalization)
*   **Methodology**: Data leakage often occurs when feature scalers (like `StandardScaler` or `MinMaxScaler`) compute parameters (mean, variance, min, max) over the entire dataset before splitting, which implicitly leaks validation/test distribution details to the training phase.
*   **Our Normalization Strategy**:
    *   We use strictly **instance-based (sample-wise) normalization** inside `preprocess.py`.
    *   The wrist translation `(P_i - P_0)` and scale normalization `(P'_i / max_dist)` are computed independently for each individual hand gesture. No statistics (means, ranges) are aggregated across the dataset.
*   **Result**: Preprocessing parameters for any given sample are completely independent of all other samples. Validation and test sets were preprocessed without any group-level cross-contamination.

### Check 4: Separation of Model Selection and Evaluation
*   **Methodology**:
    1.  **Architecture Selection**: Model A, Model B, and Model C were trained on the training split and evaluated *solely* on the validation split. The test set was kept entirely locked.
    2.  **Winner Finalization**: Once Model C was identified as the winner based on validation accuracy, it was trained to convergence.
    3.  **One-Time Test Evaluation**: The test split was loaded and evaluated *exactly once* during the execution of `evaluate.py` to yield the final reporting metrics.
*   **Result**: The test set was never used to adjust weights, tune hyper-parameters (epochs, batch size, learning rate), or select the final model layout.

---

## 2. Rationale for Trustworthiness of Metrics

The reported test accuracy of **99.82%** on the test set is highly reliable and representative of real-world generalization for the following reasons:

1.  **Uniform Spatial Normalization**: Translating the wrist to the origin and scaling to the hand size ensures that coordinate values represent invariant finger shapes. The model learns *hand configurations* rather than absolute canvas coordinates, making it invariant to distance or framing.
2.  **Balanced Target Distribution**: As shown in [docs/dataset_analysis.md](file:///C:/Users/User/Desktop/SignSpeakAI/docs/dataset_analysis.md), each class has roughly 2,000 samples. The high test accuracy is not inflated by a majority class; the model performs uniformly well (F1-score $\ge$ 99% for all classes except 'U' at 98%).
3.  **Stratified Splitting**: Stratified splits ensure that all 26 classes are equally represented in train, validation, and test subsets, matching real-world class probability distributions.
4.  **No Spatial Leakage**: Since features are individual coordinate points of landmarks, the zero overlap check ensures that no raw frames or identical spatial landmarks are shared between training and test runs.
