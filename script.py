import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectFromModel
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def prepare_data(i_mrna, i_label) -> pd.DataFrame:
    """Prepares and merges mRNA and clinical data."""
    try: 
        logging.info(f"Loading data from {i_mrna} and {i_label}...")
        mrna_df = pd.read_csv(i_mrna, sep='\t', comment='#')
        cli_spl_df = pd.read_csv(i_label, delimiter='\t', comment='#')
        logging.info(f"mRNA shape: {mrna_df.shape}, cli_spl shape: {cli_spl_df.shape}")

        logging.info(f"mRNA head:")
        print(mrna_df.head(2))
        logging.info(f"clic_spl head:")
        print(cli_spl_df.head(2))
        
        label_name = cli_spl_df[['ONCOTREE_CODE', 'CANCER_TYPE_DETAILED']].groupby('ONCOTREE_CODE').agg(lambda x: x.unique())
        logging.info(f"Label name:")
        print(f"{label_name}")

        label_extract = cli_spl_df[['SAMPLE_ID','ONCOTREE_CODE']] 
        logging.info(f"Label extract shape: {label_extract.shape}")
        logging.info(f"Label extract head:")
        print(label_extract.head(2))

        logging.info(f"Transposing the mRNA dataset for merging...")
        mrna_df['SAMPLE_ID'] = mrna_df['Entrez_Gene_Id'].copy()
        mrna_df.set_index(mrna_df.iloc[:,-1], inplace=True)
        mrna_df = mrna_df.drop(columns=['Hugo_Symbol', 'Entrez_Gene_Id', 'SAMPLE_ID'])
        mrna_trans = mrna_df.transpose()

        logging.info(f"Merging the datasets...")
        mrna_trans.reset_index(inplace=True)
        mrna_trans.rename(columns={'index': 'SAMPLE_ID'}, inplace=True)
        merged_df = pd.merge(mrna_trans, label_extract, on = 'SAMPLE_ID', how = 'inner')
        logging.info(f"Merged dataset shape: {merged_df.shape}")
        logging.info(f"Merged dataset:")
        print(merged_df.head(2))
        logging.info(f"Prediction label distribution:") 
        print(f"{merged_df['ONCOTREE_CODE'].value_counts()}")
        
        return merged_df

    except Exception as e:
        logging.error(f"Error in prepare_data: {e}")
        raise

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Cleans data by removing missing values and renaming duplicate columns."""
    try: 
        logging.info("Cleaning data...")
        logging.info(f"Data shape before dropping missing values: {data.shape}")
        logging.info(f"Removing columns with missing values...")
        data.dropna(axis=1, inplace=True)
        logging.info(f"Data shape after dropping missing values: {data.shape}")

        logging.info("Renaming duplicate columns...") 
        """column names are similar but they have different expressing values, so we rename instead of dropping them"""
        counts = {}
        new_columns = []
        for col in data.columns:
            if col not in counts:
                counts[col] = 1
                new_columns.append(col)
            else:
                counts[col] += 1
                new_columns.append(f"{col}_{counts[col]}")
        data.columns = new_columns

        return data
    
    except Exception as e:
        logging.error(f"Error in clean_data: {e}")
        raise

def predict_metrics(model, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> dict:
    """Evaluates model performance using various metrics."""
    try: 
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        auc_macro =roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
        auc_micro = roc_auc_score(y_test, y_proba, multi_class='ovr', average='micro')
        accuracy = accuracy_score(y_test, y_pred)
        precision_micro = precision_score(y_test, y_pred, average='micro', zero_division=0)
        recall_micro = recall_score(y_test, y_pred, average='micro')
        f1_micro = f1_score(y_test, y_pred, average='micro')
        return {
            'AUC_macro': auc_macro,
            'AUC_micro': auc_micro,
            'Accuracy': accuracy,
            'Precision': precision_micro,
            'Recall': recall_micro,
            'F1': f1_micro
        }
        
    except Exception as e:
        logging.error(f"Error in predict_metrics: {e}")
        raise

def process_file(data: pd.DataFrame, kfold: int, model, selection_method, skip_ft_selection: bool, smote: bool, scaler: bool) -> pd.DataFrame:
    """Processes data with k-fold cross-validation and returns average metrics."""
    try:
        logging.info(f"Processing dataset...")
        logging.info(f"Data shape: {data.shape}")
        logging.info(f"Heading of dataset:")
        print(data.head())
        
        data.columns = data.columns.astype(str)

        X = data.iloc[:,1:-1]
        y = data.iloc[:, -1]

        le = LabelEncoder()
        y = le.fit_transform(y)

        skf = StratifiedKFold(n_splits=kfold, random_state=42, shuffle=True)
        metrics_results = []

        for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

            if scaler:
                logging.info("Normalize features...")
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_train_scaler = scaler.fit_transform(X_train)
                X_test_scaler = scaler.transform(X_test)
                feature_names = X_train.columns.tolist()
                X_train = pd.DataFrame(X_train_scaler, columns=feature_names)
                X_test = pd.DataFrame(X_test_scaler, columns=feature_names)

            if smote:
                from imblearn.over_sampling import SMOTE
                logging.info(f"Oversampling with SMOTE...")
                smote = SMOTE(sampling_strategy='auto', random_state=42)
                X_train, y_train = smote.fit_resample(X_train, y_train)

            if not skip_ft_selection:
                logging.info(f"Selecting features... This may take a while...")
                selection_method.fit(X_train, y_train)
                importances = selection_method.feature_importances_
            
                non_zero_importances = importances[importances > 0]
                mean = np.mean(non_zero_importances)
                selector = SelectFromModel(selection_method, prefit=True, threshold=mean)
                X_train = selector.transform(X_train)
                X_test = selector.transform(X_test)

            else:
                logging.info("Skipping feature selection...")

            metrics = predict_metrics(model, X_train, X_test, y_train, y_test)
            metrics_results.append(metrics)
            logging.info(f"Fold {fold + 1} AUC_macro: {metrics['AUC_macro']:.4f}")
            logging.info(f"Fold {fold + 1} AUC_micro: {metrics['AUC_micro']:.4f}")

        avg_metrics = pd.DataFrame(metrics_results).mean()
        logging.info(f"AUC_macro: {avg_metrics['AUC_macro']:.4f}")
        logging.info(f"AUC_micro: {avg_metrics['AUC_micro']:.4f}")

        return avg_metrics
    
    except Exception as e:
        logging.info(f"Error processing: {e}")

def map_model(model):
    if model == 'logreg':
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=50000, class_weight='balanced')
    elif model == 'hisgradboost':
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(random_state=42)
    elif model == 'svm':
        from sklearn.svm import SVC
        return SVC(probability=True, class_weight='balanced')
    elif model == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=10000, class_weight='balanced')
    else:
        raise ValueError(f"Unsupported model {model}. Please select logreg or hisgradboost or svm")

def map_feature_selection(select_method):
    if select_method == 'ada':
        from sklearn.ensemble import AdaBoostClassifier
        return AdaBoostClassifier()
    elif select_method == 'xgb':
        from xgboost import XGBClassifier
        return XGBClassifier(class_weight='balanced', use_label_encoder=False, eval_metric='logloss')
    elif select_method == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(class_weight='balanced', n_estimators=5000, max_depth=300, random_state=42)
    
    else:
        raise ValueError(f"Selection method {select_method} not supported. Please select ada, xgb, or rf")
    
def main(i_mrna, i_label, kfold, model, select_method, skip_ft_selection, smote, scaler):
    data = prepare_data(i_mrna, i_label)
    data = clean_data(data)
    model = map_model(model)
    selection_method = map_feature_selection(select_method)
    avg_metrics = process_file(data, kfold, model, selection_method, skip_ft_selection, smote, scaler)
    logging.info("Prediction results:")
    print(f"{avg_metrics}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predicting Colon Cancer Types based on mRNA expression") 
    parser.add_argument('-i_mrna', type=str, required=True, help='mRNA expresison dataset') 
    parser.add_argument('-i_label', type=str, required=True, help='Clinical sample with label type dataset')
    parser.add_argument('-kfold', type=int, default=10, help='Number of folds (default=10)')
    parser.add_argument('-model', type=str, default='hisgradboost', help='Model name (defaul=hisgradboost). Support logreg, hisgradboost, svm, rf.')
    parser.add_argument('-select_method', type=str, default='rf', help='Feature selection method (default=rf). Support ada, xgb, rf.')
    parser.add_argument('--skip_ft_selection', action='store_true', help='Skip feature selection process')
    parser.add_argument('--smote', action='store_true', help='Apply SMOTE to handle class imbalance')
    parser.add_argument('--scaler', action='store_true', help='Apply MinMaxScaler to normalize features')
    args = parser.parse_args()
    
    main(args.i_mrna, args.i_label, args.kfold, args.model, args.select_method, args.skip_ft_selection, args.smote, args.scaler)