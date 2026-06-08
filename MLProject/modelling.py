import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Atur tracking URI ke server MLflow lokal
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("eksperimen_model_lokal")

# Autologging aktif
mlflow.sklearn.autolog()

# Pastikan file credit_scoring_clean.csv ada di folder yang sama
df = pd.read_csv('credit_scoring_clean.csv')
X = df.drop('class', axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run():
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    print("Model Basic berhasil di-training dan dicatat secara LOKAL di MLflow.")