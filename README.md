# Deep Learning for Tabular Data (2025 Update)

This project demonstrates how Deep Learning techniques can be effectively applied to tabular data, offering a competitive alternative to traditional machine learning models like Gradient Boosting. The code has been significantly updated in 2025 to incorporate modern best practices, performance optimizations, and a more robust experimental setup.

## Project Overview

The core of this project is to predict employee access needs based on the [Amazon Employee Access Challenge](https://www.kaggle.com/c/amazon-employee-access-challenge) dataset. It provides a side-by-side comparison of a fine-tuned XGBoost model and a Deep Neural Network built with Keras 3 (using a PyTorch backend).

## Key Features (2025 Version)

The latest version includes substantial improvements over the original implementation:

*   **Modernized Tech Stack**: Upgraded to Keras 3 with a PyTorch backend, supporting CPU, CUDA, and Apple's MPS for acceleration.
*   **Corrected Cross-Validation**: Fixed a state-leakage bug in the K-Fold cross-validation loop to establish a reliable and realistic performance baseline.
*   **Hyperparameter Optimization**: Integrated **Optuna** to perform systematic hyperparameter tuning for the XGBoost model, with the best parameters saved in the script.
*   **Advanced DNN Architecture**: The neural network has been completely refactored for better performance and stability:
    *   **Architecture**: Changed from a simple `256 -> 256` structure to a more effective `512 -> 256` funnel architecture.
    *   **Regularization**: Added `BatchNormalization` layers and increased the dropout rate to `0.4` to prevent overfitting.
    *   **Intelligent Embeddings**: Updated the embedding size calculation to `min(50, (num_categories + 1) / 2)` for better representation of categorical features.
*   **Dependency Management**: All dependencies are now correctly listed in `requirements.txt` and `pyproject.toml`.
*   **Code Quality**: Resolved all `UserWarning`s from libraries for a cleaner execution experience.

## Performance Improvements

The architectural and methodological improvements resulted in a significant performance boost for the DNN model:

*   **Mean AUC**: Increased from `0.790` to **`0.809`**.
*   **Stability**: Cross-fold score standard deviation drastically reduced from `0.125` to **`0.022`**, indicating a much more reliable model.

## Getting Started

### Prerequisites

- Python 3.12+
- An active virtual environment (e.g., using `venv` or `conda`)

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/lmassaron/deep_learning_for_tabular_data.git
    cd deep_learning_for_tabular_data
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### How to Run

To run the full experiment, including training the XGBoost and DNN models with 5-fold cross-validation and generating submission files, execute the main script:

```bash
python run_experiment.py
```

The script will output the cross-validation scores for both models and save the predictions to `tabular_dnn_submission.csv` and `xgboost_submission.csv`.

---

## Legacy Version (2019-2020)

The original version of this project was presented at GDG Venezia in 2019 and featured in presentations in 2020. It demonstrated how to achieve good results using TensorFlow/Keras integrated with Scikit-learn and Pandas.

### Original Workshop Code on Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lmassaron/deep_learning_for_tabular_data/blob/master/deep-learning-for-tabular-data.ipynb)

### Tutorial on YouTube (GDG Venezia 2019)
<a href="https://www.youtube.com/watch?v=nQgUt_uADSE"><img src="./GDG_Venezia_2019.PNG" alt="GDG Venezia 2019" height="320px" align="center"></a>

https://www.youtube.com/watch?v=nQgUt_uADSE