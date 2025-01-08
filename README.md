## Multi-label classification task - Predicting colon cancer based on mRNA expression

This script is designed to predict colon cancer types using mRNA expression and clinical sample data. The process involves data preprocessing, feature selection, and classification using various machine learning models.

### Features

1. Data Preparation:
    - Load data, extract label as outcome prediction. 
    - Merges mRNA expression data with clinical sample labels.
    - Handles missing values and renames duplicate columns.

2. Feature Selection:
    - Supports multiple feature selection methods (Random Forest, XGBoost, AdaBoost).

3. Classification Models:
    - Logistic Regression, HistGradientBoost, SVM, and Random Forest.

4. Cross-Validation:
    - Performs Stratified K-Fold cross-validation for robust evaluation.

5. Optional Preprocessing:
    - SMOTE: Handles class imbalance through oversampling.
    - Scaling: Applies StandardScaler for feature normalization.

6. Metrics:
    - Evaluates performance using AUC (macro and micro), accuracy, precision(micro), recall (micro), and F1-score (micro).

### Required Libraries:
- Python 3.x
- pandas
- numpy
- scikit-learn
- imblearn
- xgboost

Install dependencies using: `pip install pandas numpy scikit-learn imbalanced-learn xgboost`

### Usage
Run the script using the command line:
```
python script.py -i_mrna <mRNA_dataset_path> -i_label <clinical_labels_path> [-kfold <num_folds>] [-model <model_name>] [-select_method <selection_method>] [--skip_ft_selection] [--smote] [--scaler]
```
```
Arguments:
-i_mrna (str, required): Path to the mRNA dataset file.
-i_label (str, required): Path to the clinical labels file.
-kfold (int, optional): Number of K-Folds for cross-validation (default: 10).
-model (str, optional): Model for classification (logreg, hisgradboost, svm, rf) (default: hisgradboost).
-select_method (str, optional): Feature selection method (ada, xgb, rf) (default: rf).
--skip_ft_selection (flag, optional): Skip feature selection. Not recommended due to large datasets. 
--smote (flag, optional): Apply SMOTE for class imbalance.
--scaler (flag, optional): Apply StandardScaler for normalization.
```
Example:
```
python script.py -i_mrna data/data_mrna_seq_v2_rsem.txt -i_label data/data_clinical_sample.txt -kfold 5 -model hisgradboost -select_method rf --scaler
```

### Workflow
1. Prepare Data:
    - Load and merge mRNA and clinical datasets.
    - Transpose mRNA dataset and align with clinical labels.

2. Clean Data:
    - Remove missing values.
    - Rename duplicate columns.

3. Feature Selection (Recommended):
    - Use selected method to retain the most important features.

4. Preprocessing:
    - Apply SMOTE for oversampling.
    - Normalize features using StandardScaler.

5. Train and Evaluate:
    - Perform Stratified K-Fold cross-validation.
    - Evaluate metrics (AUC, accuracy, precision, recall, F1-score).

### Output
- Logs the progress and intermediate results to the console.
- Prints performance metrics for each fold and overall averages.

### File Structure
```
project_colon_cancer/
├── data/
│   ├── data_mrna_seq_v2_rsem.txt       # mRNA dataset (example)
│   ├── data_clinical_sample.txt        # Clinical labels (example)
├── script.py                           # Main script
└── README.md                           # Documentation
```
### Notes

1. Ensure the input datasets are properly formatted with matching sample IDs. Please modify the script if needed corresponding to your data. 

2. Large files (>50MB) may require Git LFS for version control.

3. Modify hyperparameters in the script for better results on specific datasets.

### Contributions
Contributions are welcome! Please open an issue or submit a pull request on GitHub. 

Many thanks! 
