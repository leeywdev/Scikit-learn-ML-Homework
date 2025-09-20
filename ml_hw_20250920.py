### 202135768 이유원 ###

# Dataset 다운로드
import kagglehub

# Download latest version
path = kagglehub.dataset_download("ninjacoding/breast-cancer-wisconsin-benign-or-malignant")

print("Path to dataset files:", path)

# 라이브러리 설치
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# 데이터 로드 함수
def load_data(filepath):
    # 데이터셋 추출
    dataset = pd.read_csv('../.cache/kagglehub/datasets/ninjacoding/breast-cancer-wisconsin-benign-or-malignant/versions/3/tumor.csv')

    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    #print(X)
    #print(y)
    return X, y

# 테스트 데이터셋과 트레이닝 데이터셋 분류 (0.3)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

# 특징 스케일링
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 의사결정 트리 엔트로피
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state=0)
classifier.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
CM = confusion_matrix(y_test, y_pred)
print(CM)
print("Decision Tree with Entropy: ", accuracy_score(y_test, y_pred))

# 의사결정 트리 지니 지수
classifier = DecisionTreeClassifier(criterion = 'gini', random_state=0)
classifier.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
CM = confusion_matrix(y_test, y_pred)
print(CM)
print("Decision Tree with Gini index: ", accuracy_score(y_test, y_pred))
