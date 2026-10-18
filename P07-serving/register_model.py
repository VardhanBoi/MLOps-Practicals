
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

# Use a database-backed MLflow tracking store
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("P07-Delivery-Time")

# Load the model you already trained in P07.ipynb
model_path = Path("work/model.joblib")

if not model_path.exists():
    raise FileNotFoundError(f"Model not found: {model_path.resolve()}")

model = joblib.load(model_path)
model_name = "DeliveryTimePredictor"

# Log the existing model and register it
with mlflow.start_run() as run:
    mlflow.sklearn.log_model(
        sk_model=model,
        name="delivery_time_model",
        registered_model_name=model_name,
    )
    run_id = run.info.run_id

client = MlflowClient()

# Find the registered version created for this run
versions = client.search_model_versions(
    f"name='{model_name}'"
)
matching = [v for v in versions if v.run_id == run_id]

if not matching:
    raise RuntimeError("Model was logged but no registered version was found.")

version = max(matching, key=lambda v: int(v.version))

# Promote this version to Production
client.transition_model_version_stage(
    name=model_name,
    version=version.version,
    stage="Production",
    archive_existing_versions=True,
)

print("Registration successful")
print("Model name:", model_name)
print("Model version:", version.version)
print("Stage: Production")
print("Run ID:", run_id)
