# Preprocessing Pipeline Documentation

This document describes the normalization strategy, label encoding, and dataset splitting applied in the preprocessing stage of the SignSpeak AI ML pipeline.

---

## 1. Normalization Strategy

Hand landmark coordinates returned by MediaPipe can vary significantly based on:
1.  **Distance**: How far the hand is from the camera lens.
2.  **Location**: Where the hand is positioned within the video frame.
3.  **Physical hand dimensions**: Differences in user hand sizes.

To achieve robust real-time translation invariant to translation, scale, and hand size, we apply a two-step mathematical normalization:

### Translation Normalization
For each active hand detected, we shift all 21 landmark points so that the wrist landmark (point 0) resides at the coordinate system origin `(0, 0, 0)`:

$$P'_i = P_i - P_0 \quad \text{for } i \in [0, 20]$$

Where:
*   $P_i$ is the original 3D coordinate vector $(x_i, y_i, z_i)$ of landmark $i$.
*   $P_0$ is the coordinate vector of the wrist landmark (index 0).
*   $P'_i$ is the translated coordinate vector.

This step strips away the spatial position of the hand, making the features location-invariant.

### Scale Normalization
After translating, we scale all coordinates by the hand size. We define hand size as the maximum Euclidean distance from the wrist (which is now at the origin) to any other landmark:

$$S = \max_{i \in [0, 20]} \left( \| P'_i \|_2 \right)$$

$$P''_i = \frac{P'_i}{S} \quad \text{for } i \in [0, 20]$$

This projects all coordinates into a unit sphere of radius 1, bounding coordinate values to $[-1.0, 1.0]$ and making features distance-invariant.

---

## 2. Handling Missing Hands (-1.0 Padding)

In single-handed gestures (which represents **23.39%** of the dataset), the inactive hand is padded with **`-1.0`** values.
*   **Safety check**: During preprocessing, we check if the first coordinate (wrist X) is exactly `-1.0` (or if all coordinates are `-1.0`).
*   **Preservation**: If a hand is identified as missing, we **bypass** translation and scaling. The coordinates are preserved as `-1.0` vectors, ensuring the neural network recognizes the absence of the hand without mathematical distortions.

---

## 3. Label Encoding

The target column in the raw dataset contains class numbers from `0` to `25` (representing alphabets A to Z).
*   We fitted a `scikit-learn` `LabelEncoder` to transform these label strings into numerical class indices.
*   The encoder has been saved to the designated model artifact path: `ml/model/label_encoder.pkl`. This allows the API backend to decode numerical predictions back to alphabet strings.

---

## 4. Train, Validation, and Test Splitting

We perform a stratified split to ensure each set contains an identical class distribution, preventing bias. The dataset is split into **80% Training**, **10% Validation**, and **10% Testing** sets:

| Dataset Split | Sample Count | Percentage | Purpose |
| :---: | :---: | :---: | --- |
| **Training Set** | 40,687 | 80.0% | Model parameter updates during training. |
| **Validation Set** | 5,086 | 10.0% | Hyperparameter tuning and overfitting evaluation. |
| **Testing Set** | 5,086 | 10.0% | Final generalization evaluation (unseen data). |
| **Total** | **50,859** | **100.0%** | |

The preprocessed features ($X$) have shape `(N, 127)`, consisting of:
*   `uses_two_hands` (index 0, shape `(1,)`)
*   `left_hand` normalized landmarks (indices 1-63, shape `(63,)`)
*   `right_hand` normalized landmarks (indices 64-126, shape `(63,)`)

All split arrays are serialized and saved under `dataset/preprocessed_data.npz` to optimize model training cycles in the next phase.
