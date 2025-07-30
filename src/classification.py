from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import streamlit as st

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)
    return acc, report

def show_confusion_matrix(y_true, y_pred, labels=[0, 1]):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Female", "Male"])

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap="Blues", ax=ax, values_format='d')  # force normal decimal format
    st.pyplot(fig)
    plt.clf()



def tune_model(model_name, X_train, y_train):
    if model_name == "Random Forest":
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5]
        }
        model = RandomForestClassifier()
    elif model_name == "SVM":
        param_grid = {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"]
        }
        model = SVC(probability=True)
    else:
        raise ValueError("Unsupported model for tuning")

    grid = GridSearchCV(model, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)

    print(f"Best Parameters for {model_name}: {grid.best_params_}")
    return grid.best_estimator_