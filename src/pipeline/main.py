import datetime
import uuid

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn import set_config

import flama

set_config(display="diagram")

# Loading data:
dataset = pd.read_parquet("data/data.parquet")

X = dataset.drop(columns=["Exited"]).values

y = dataset["Exited"].values

columns = dataset.columns

# Preprocessing numerical features:
numeric_transformer = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)
numeric_features = [
    columns.get_loc(c)
    for c in dataset.select_dtypes(include=["int64", "float64"])
    .drop(["RowNumber", "CustomerId", "Exited"], axis=1)
    .columns.values
]

# Preprocessing categorical features:
categorical_transformer = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)
categorical_features = [
    columns.get_loc(c)
    for c in dataset.select_dtypes(include=["object"])
    .drop(["Surname"], axis=1)
    .columns.values
]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
preprocessor = ColumnTransformer(
    [
        ("numerical", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features),
    ]
)

# Model train:
mlp = MLPClassifier(
    hidden_layer_sizes=(8, 6, 1),
    max_iter=300,
    activation="tanh",
    solver="adam",
    random_state=123_456,
)
pipeline = Pipeline(
    [
        ("preprocessing", preprocessor),
        ("mlp_classifier", mlp),
    ]
)
pipeline.fit(X_train, y_train)

# Model evaluation:
pipeline.score(X_test, y_test)
pipeline.predict(X_test)

# Model dump:
flama.dump(
    pipeline,
    "data/model.flm",
    model_id=uuid.UUID("e9d4a470-1eb2-423e-8fb3-eaf236158ab3"),
    timestamp=datetime.datetime(2023, 3, 10, 11, 30, 0),
    params={"solver": "adam", "random_state": 123_456, "max_iter": 300},
    metrics={
        "roc_auc_score": roc_auc_score(y_test, pipeline.predict(X_test)),
        "f1_score": f1_score(y_test, pipeline.predict(X_test)),
    },
    extra={
        "model_author": "Vortico",
        "model_description": "Churn classifier",
        "model_version": "1.0.0",
        "tags": ["loss", "churn"],
    },
    artifacts={"artifact.json": "./data/artifact.json"},
)

# Load the pipeline:
p_artifact = flama.load("./data/model.flm")
l_pipeline = p_artifact.model

