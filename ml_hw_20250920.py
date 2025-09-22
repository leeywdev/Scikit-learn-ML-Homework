### 202135768 이유원 ###

# 라이브러리 설치
import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix
from os import system

# Download latest version Kaggle Dataset
path = kagglehub.dataset_download("ninjacoding/breast-cancer-wisconsin-benign-or-malignant")
print("Path to dataset files:", path)

# 데이터 로드 함수
def load_data(filepath):
    dataset = pd.read_csv(filepath)

    X = dataset.iloc[:, 1:-1]
    y = dataset.iloc[:, -1].replace({2:0, 4:1}) # 0: benign(양성), 1: malignant(악성)

    return X, y

def run(X, y, cv_list=[5,10,15]):

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
        'dt_entropy': {
            'clf__max_depth':  [3, 5, None], # 트리 최대 깊이
            'clf__min_samples_split': [2, 5, 10], # 내부 노드 분할 최소 샘플 수
        },
        'dt_gini': {
            'clf__max_depth':  [3, 5, None], # 트리 최대 깊이
            'clf__min_samples_split': [2, 5, 10], # 내부 노드 분할 최소 샘플 수
        },
        'logreg': {
            'clf__C': [0.1, 1, 10, 100], # 정규화 파라미터
            'clf__penalty': ['l2'], # 정규화 종류, 'l1'은 'liblinear' 또는 'saga' solver에서만 지원
            'clf__solver': ['liblinear', 'lbfgs'] # 최적화
        },
        'svm': {
            'clf__C': [0.1, 1, 10, 100], # 정규화 파라미터
            'clf__kernel': ['linear', 'rbf'], # 커널
            'clf__gamma': ['scale','auto'], # ‘rbf’ 계수
        }
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
                grid = GridSearchCV(pipe, param_grid[model_name], cv=cv,scoring='accuracy', n_jobs=-1)
                grid.fit(X, y)
                best_score = grid.best_score_
                best_params = grid.best_params_

                # Confusion matrix (전체 데이터에 대해 예측)
                y_pred = grid.predict(X)
                cm = confusion_matrix(y, y_pred)

                results.append({
                    'model': model_name,
                    'scaler': scaler_name,
                    'K-fold': cv,
                    'best_params': best_params,
                    'best_score': best_score,
                    'cm': cm
                })
    return pd.DataFrame(results)

def print_user_manual():
    print("""
    Usage:
    1. Prepare the Wisconsin Cancer Dataset as a CSV file.
    2. Call load_data(filepath) to load the data.
    3. Call run_experiments(X, y, cv_list=[5, 10, 15]) to run all experiments.
    4. The function returns a DataFrame summarizing the best accuracy and parameters for each combination.
    """)

# def viz(clf):
#     dot_data = tree.export_graphviz(clf,   # 의사결정나무 모형 대입
#                                out_file = None,  # file로 변환할 것인가
#                                feature_names = iris.feature_names,  # feature 이름
#                                class_names = iris.target_names,  # target 이름
#                                filled = True,           # 그림에 색상을 넣을것인가
#                                rounded = True,          # 반올림을 진행할 것인가
#                                special_characters = True)   # 특수문자를 사용하나

# graph = graphviz.Source(dot_data)        


if __name__ == "__main__":
    print_user_manual()
    X, y = load_data('../.cache/kagglehub/datasets/ninjacoding/breast-cancer-wisconsin-benign-or-malignant/versions/3/tumor.csv')
    results = run(X, y, cv_list=[5, 10, 15])
    print("Results:\n", results)
    best_by_model = results.loc[results.groupby('model')['best_score'].idxmax()]
    print("Best Results by Model:\n", best_by_model[['model', 'scaler', 'K-fold', 'best_params', 'best_score']])
    
    # 각 모델에 대해 혼동행렬 출력
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,10))
    axes = axes.flatten()
    for ax, (_, row) in zip(axes, best_by_model.iterrows()):
        cm = row['cm']                      # run()에서 저장한 cm
        model_name = row['model']
        scaler_name = row['scaler']

        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
        ax.set_title(f"{model_name} (scaler: {scaler_name})")

    plt.tight_layout()  
    plt.show()
