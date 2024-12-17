# Extending XReason for Efficient Formal XAI and Adversarial Detection

This repository contains the code and resources for the project **Extending XReason for Efficient Formal XAI and Adversarial Detection**, which presents an extension to the XReason tool. For details on the original XReason tool, please refer to the [XReason GitHub repository](https://github.com/alexeyignatiev/xreason/).


## Features
- **Support for LightGBM & XGBoost**: Works with tabular data.
- **Formal Explanations**: Instance and class-level explanations using a MaxSAT solver.
- **Adversarial Generation**: Generating adversarial examples based on formal explanations.
- **Adversarial Detection**: Identifying adversarial samples based on formal explanations.


## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Folder Structure](#Folder-Structure)
- [Key Files](#Key-Files)
- [Usage](#usage)
- [Methodology](#methodology)

## Installation
To run this project, you'll need to install the required dependencies.

1. Clone the repository:

    ```bash
    git clone https://github.com/amirajemaa/Extending-XReason-for-Efficient-Formal-XAI-and-Adversarial-Detection.git
    cd Extending-XReason-for-Efficient-Formal-XAI-and-Adversarial-Detection
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Ensure you have Jupyter Notebook installed:

    ```bash
    pip install notebook
    ```
## Dataset

This project uses a customized version of the [CICIDS-2017 dataset](https://www.unb.ca/cic/datasets/ids-2017.html), which is related to network security and intrusion detection.

## Folder Structure

- **`CICIDS_dataset/`**: Contains the customized CICIDS-2017 dataset used in the project.
- **`CICIDS_results/`**: Stores any exported data files from analysis notebooks.
- **`src/`**: Source Folder
- **`xgbooster/`**: Files related to original Xreason.

## Key Files

- **`train.py`**: Script for training LightGBM.
- **`encoder.py`**: Provides encoding functions of LightGBM model.
- **`explainer.py`**: Functions for formal instance explanation.
- **`heuristic.py`**: Functions for comparing heuristic methods with formal explanations.
- **`class_explanation.py`**: Defines class-based explanation generation, providing intervals for most important features for each class.
- **`adversarial.py`**: Functions for generating and detecting adversarial examples using formal explanations.
- **`data.py`**: Script of original XReason.
- **`mxreason.py`**: Script of original XReason.
- **`options.py`**: Script of original XReason.
- **`xreason.py`**: Script of original XReason.
- **`erc2.py`**: Script of original XReason.
- **`CICIDS-2017_Dataset_Formal_explanations.ipynb`**: Notebook for generating instance formal explanations and class-level formal explanations generated for LightGBM models on the CICIDS-2017 dataset.
- **`CICIDS-2017_Dataset_heuristic_vs_formal_explanations.ipynb`**: Compares heuristic explanations (e.g., SHAP, LIME) with formal explanations in terms of robustness and correctness.
- **`CICIDS-2017_Dataset_Adversarial_unit.ipynb`**: Notebook for generating adversarial samples from the CICIDS-2017 dataset and evaluating model robustness against these examples.

## Usage

1. Launch the Jupyter Notebook server:

    ```bash
    jupyter notebook
    ```

2. Open the notebooks:
    - **[CICIDS-2017_Dataset_Formal-explanations.ipynb](src/CICIDS-2017_Dataset_Formal_explanations.ipynb)**: Explains LightGBM model predictions using instance and class-level formal explanations.
    - **[CICIDS-2017_Dataset_heuristic_vs_formal_explanations.ipynb](src/CICIDS-2017_Dataset_heuristic_vs_formal_explanations.ipynb)**: Demonstrates heuristic vs. formal explanations for the CICIDS-2017 dataset.
    - **[CICIDS-2017_Dataset_Adversarial_unit.ipynb](src/CICIDS-2017_Dataset_Adversarial_unit.ipynb)**: Notebook for generating adversarial samples from the CICIDS-2017 dataset and evaluating model robustness against these examples.
4. Follow the instructions within each notebook to run the experiments.

## Methodology

The project extends the XReason tool by introducing adversarial detection and generation, leveraging formal instance-based and class-based explanations. These explanations help identify critical features that impact model decisions and evaluate robustness against adversarial attacks.
![](meth.png?raw=true "Methodology")

For more details, please refer to the [project page](https://hvg.ece.concordia.ca/projects/fvai/pr2).
