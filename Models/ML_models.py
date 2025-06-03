import numpy as np
import pandas as pd
import os
import joblib
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, matthews_corrcoef
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import scale
import ast
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, roc_auc_score, average_precision_score
from tqdm import tqdm

def str_to_list(s):
    return ast.literal_eval(s)

def MLPredModel(X, Y, bacteria, mode, encoding, method, drug_name, Normalization=True, seed=7, save_res='easy', ratio=(1, 1)):
    print(method)
    MODEL = {
        'LR': LogisticRegression(),
        'RF': RandomForestClassifier(),
        'SVM': SVC()
    }
    if encoding == 'FCGR':
        X = X.reshape(X.shape[0], -1)
    if Normalization:
        X = scale(X)
    
    param_grid = {
        "LR": {
            'C': [0.1, 1, 5, 10],
            'max_iter': [5000]
        },
        "RF": {
            'n_estimators': [50, 100],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        },
        "SVM": {
            'C': [1, 3, 5],
            'kernel': ['linear', 'sigmoid'],
            'gamma': ['scale', 'auto']
        }
    }
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    model = MODEL[method]
    grid_search = GridSearchCV(model, param_grid[method], cv=2, scoring='accuracy')
    print(f"{bacteria}, {drug_name}, Training {method} model ...")
    
    grid_search.fit(x_train, y_train)
    best_model = grid_search.best_estimator_
    folder_path = 'models'
    if not os.path.exists(f'{folder_path}/{drug_name}/'):
        os.makedirs(f'{folder_path}/{drug_name}/')
    model_filename = f'{folder_path}/{drug_name}/{method}_{ratio[1]}_model.pkl'
    joblib.dump(best_model, model_filename)

    preds = best_model.predict(x_test)
    cm = confusion_matrix(y_test, preds)
    
    MCC = matthews_corrcoef(y_test, preds)
    precision = precision_score(y_test, preds, average='macro')
    recall = recall_score(y_test, preds, average='macro')
    f1 = f1_score(y_test, preds, average='macro')
    acc = accuracy_score(y_test, preds)
    AUROC = roc_auc_score(y_test, preds)
    AUPRC = average_precision_score(y_test, preds)
    
    return [MCC, precision, recall, f1, acc, AUROC, AUPRC], cm

if __name__ == '__main__':
    Methods = ['LR', 'RF', 'SVM']
    Encoding = 'Label_Encoding'
    save_res = 'easy'
    seed = [34]
    Drug_list = ['AMP', 'AUG', 'AXO', 'CHL', 'CIP', 'COT', 'FIS', 'FOX', 'NAL', 'STR', 'TET']
    positive_negative_ratios = [(1, 1), (1, 2), (1, 3), (1, 4)]
    res_all_drug = []
    
    for drug in tqdm(Drug_list, desc='Drugs'):
        for ratio in tqdm(positive_negative_ratios, desc='Ratios', leave=False):
            snp_seq = pd.read_csv(f'snps.csv')
            snp_seq['Seq_mapping'] = snp_seq['Seq_mapping'].apply(str_to_list)
            X = np.array(snp_seq['Seq_mapping'].tolist())
            Y = np.array(snp_seq[f'{drug} Concl'].tolist())
            for m in tqdm(Methods, desc='Methods', leave=False):
                res, cm = MLPredModel(X, Y, bacteria='bacteria', mode='ATGC', encoding=Encoding, method=m, drug_name=drug, Normalization=False, seed=seed[0], ratio=ratio)
