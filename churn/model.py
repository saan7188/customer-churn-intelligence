from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from churn.data import FEATURES, validate

def train(data, algorithm="logistic"):
    data = validate(data)
    x_train, x_test, y_train, y_test = train_test_split(data[FEATURES], data.churn, test_size=.25, random_state=42, stratify=data.churn)
    numeric = Pipeline([("fill", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    category = Pipeline([("fill", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))])
    preprocessing = ColumnTransformer([("numeric", numeric, FEATURES[:3]), ("category", category, ["contract"])])
    if algorithm not in ("logistic", "forest"):
        raise ValueError("Unknown algorithm")
    estimator = LogisticRegression(max_iter=500) if algorithm == "logistic" else RandomForestClassifier(n_estimators=100, random_state=42, min_samples_leaf=3)
    pipeline = Pipeline([("preprocess", preprocessing), ("model", estimator)])
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    metrics = {"accuracy": accuracy_score(y_test, predictions), "precision": precision_score(y_test, predictions, zero_division=0), "recall": recall_score(y_test, predictions, zero_division=0), "f1": f1_score(y_test, predictions, zero_division=0), "roc_auc": roc_auc_score(y_test, probabilities)}
    return pipeline, metrics, confusion_matrix(y_test, predictions, labels=[0,1])
