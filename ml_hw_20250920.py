### 202135768 이유원 ###

# 라이브러리 설치
import kagglehub
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score

# Download latest version Kaggle Dataset
path = kagglehub.dataset_download("ninjacoding/breast-cancer-wisconsin-benign-or-malignant")
print("Path to dataset files:", path)

# 데이터 로드 함수
def load_data(filepath):
    dataset = pd.read_csv(filepath)

    X = dataset.iloc[:, 1:-1]
    y = dataset.iloc[:, -1].replace({2:0, 4:1}) # 0: benign(양성), 1: malignant(악성)

    return X, y

def run(X, y, cv_list=[3,5,10,30]):

    """
    여러가지 분류 모델을 다양한 데이터 스케일링, 하이퍼 파라미터로 수행합니다.
    인자값(Args):
        X (pd.DataFrame): 특성 행렬(Feature Matrix)
        y (pd.Series): 타겟 벡터
        cv_list (list): k-fold cross-validation 값 목록
    결과값(Returns):
        results (pd.DataFrame): 모든 분석 결과 요약
    """

    scalers = {
        'none': None,
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'robust': RobustScaler()
    }

    models = {
        'dt_entropy': DecisionTreeClassifier(criterion='entropy'),
        'dt_gini': DecisionTreeClassifier(criterion='gini'),
        'logreg': LogisticRegression(max_iter=1000),
        'svm': SVC()
    }

    param_grid = {
        'dt_entropy': {'clf__max_depth': [3,5,10]},
        'dt_gini': {'clf__max_depth': [3,5,10]},
        'logreg': {'clf__C': [0.1, 1, 10]},
        'svm': {'clf__C': [0.1, 1, 10], 'clf__kernel': ['linear', 'rbf']}
    }

    results = []
    for model_name, model in models.items():
        for scaler_name, scaler in scalers.items():
            for cv in cv_list:
                steps = []
                if scaler is not None:
                    steps.append(('scaler', scaler))
                steps.append(('clf', model))
                pipe = Pipeline(steps)
                grid = GridSearchCV(pipe, param_grid[model_name], cv=cv,scoring='accuracy')
                grid.fit(X, y)
                best_score = grid.best_score_
                best_params = grid.best_params_
                results.append({
                    'model': model_name,
                    'scaler': scaler_name,
                    'K-fold': cv,
                    'best_params': best_params,
                    'best_score': best_score
                })
    return pd.DataFrame(results)

def print_user_manual():
    print("""
    Usage:
    1. Prepare the Wisconsin Cancer Dataset as a CSV file.
    2. Call load_data(filepath) to load the data.
    3. Call run_experiments(X, y, cv_list=[3, 5, 10, 30]) to run all experiments.
    4. The function returns a DataFrame summarizing the best accuracy and parameters for each combination.
    """)

if __name__ == "__main__":
    print_user_manual()
    X, y = load_data('../.cache/kagglehub/datasets/ninjacoding/breast-cancer-wisconsin-benign-or-malignant/versions/3/tumor.csv')
    results = run(X, y, cv_list=[3, 5, 10, 30])
    print("Results:\n", results)
    best_by_model = results.loc[results.groupby('model')['best_score'].idxmax()]
    print("Best Results by Model:\n", best_by_model[['model', 'scaler', 'K-fold', 'best_params', 'best_score']])