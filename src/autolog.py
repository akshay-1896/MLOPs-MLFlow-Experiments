import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import dagshub
dagshub.init(repo_owner='officialakshay1896', repo_name='MLOPs-MLFlow-Experiments', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/officialakshay1896/MLOPs-MLFlow-Experiments.mlflow")

# Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Define the params for RF model
max_depth = 20
n_estimators = 8

# Mention your experiment name
mlflow.autolog()
mlflow.set_experiment("MLflow_Wine_Classification")

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Creating a confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    # save the plot to a file
    plt.savefig('confusion_matrix.png')

    # save the model
    # mlflow.sklearn.save_model(rf, "Random-Forest-Model")

    # log artifacts(the plot) using mlflow
    mlflow.log_artifact(__file__)
    mlflow.log_artifacts("Random-Forest-Model", artifact_path="model")
    
    # tags
    mlflow.set_tags({"Author": "Akshay", "Project": "Wine_Classification"})

    # log the model
    # mlflow.sklearn.log_model(rf, "Random-Forest-Model")
    
    # Register the model
    # model_name = "wine_classifier"
    # mlflow.register_model(
    #     f"runs:/{mlflow.active_run().info.run_id}/random_forest_model", model_name
    # )


    print(accuracy)