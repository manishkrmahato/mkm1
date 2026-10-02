https://chatgpt.com/share/6ac01797-cca8-83ee-8660-dc97fe5f6c1e

!pip install numpy pandas matplotlib seaborn scikit-learn scipy requests beautifulsoup4 lxml html5lib openpyxl

python -m pip install numpy pandas matplotlib seaborn scikit-learn scipy requests beautifulsoup4 lxml html5lib openpyxl

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import requests
import time

from bs4 import BeautifulSoup

from scipy.stats import ttest_ind, ttest_rel

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    Ridge,
    Lasso,
    ElasticNet
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import RFE

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)
