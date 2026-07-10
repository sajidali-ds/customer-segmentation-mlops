import os
import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "customer-segmentation"
REGISTERED_MODEL_NAME = "customer-segmentation-model"


def init_mlflow():
    tracking_uri = "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    return tracking_uri 


def log_candidate_run(algorithm_name: str, params: dict, metrics: dict):
    """Log one candidate algorithm (e.g. DBSCAN) as its own nested MLflow run,
    so the full comparison is visible in the MLflow UI, not just the winner."""
    with mlflow.start_run(run_name=f"candidate-{algorithm_name}", nested=True):
        mlflow.set_tag("algorithm", algorithm_name)
        mlflow.set_tag("run_type", "candidate")
        mlflow.log_params(params)
        mlflow.log_metrics({k: v for k, v in metrics.items() if v is not None})


def log_and_register_best_model(
    model,
    algorithm_name: str,
    params: dict,
    metrics: dict,
    artifact_paths: list,
    input_example=None,
):
    """
    Logs the winning model + its artifacts under the currently active parent
    run, registers it in the Model Registry, and returns the new version
    number + run_id so the caller can decide whether to promote it.
    """
    mlflow.set_tag("algorithm", algorithm_name)
    mlflow.set_tag("run_type", "winner")
    mlflow.log_params(params)
    mlflow.log_metrics({k: v for k, v in metrics.items() if v is not None})

    for path in artifact_paths:
        if os.path.exists(path):
            mlflow.log_artifact(path)

    # Clustering models don't have .predict_proba/standard sklearn signature
    # for all algorithms (e.g. Agglomerative has no .predict), so we log via
    # the generic sklearn flavor which just pickles the fitted estimator.
    model_info = mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name=REGISTERED_MODEL_NAME,
    )

    run_id = mlflow.active_run().info.run_id
    client = MlflowClient()

    # Find the version number MLflow just created for this run
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    new_version = None
    for v in versions:
        if v.run_id == run_id:
            new_version = v.version
            break

    return new_version, run_id


def promote_if_better(new_version: str, new_silhouette: float):
    """
    Deployment gate: only promote the newly trained model to the
    'Production' alias if it beats whatever is currently in Production.
    If nothing is in Production yet, promote automatically.
    Older Production model is moved to 'Archived' instead of Production.
    """
    client = MlflowClient()

    current_prod = None
    try:
        current_prod = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "production")
    except Exception:
        current_prod = None

    if current_prod is None:
        client.set_registered_model_alias(REGISTERED_MODEL_NAME, "production", new_version)
        print(f"No existing Production model -> promoted version {new_version} to Production.")
        return True

    current_run = client.get_run(current_prod.run_id)
    current_silhouette = current_run.data.metrics.get("silhouette", -1.0)

    if new_silhouette > current_silhouette:
        client.set_registered_model_alias(REGISTERED_MODEL_NAME, "production", new_version)
        client.set_registered_model_alias(REGISTERED_MODEL_NAME, "archived", current_prod.version)
        print(
            f"New model (silhouette={new_silhouette:.4f}) beats current Production "
            f"(silhouette={current_silhouette:.4f}) -> promoted version {new_version}."
        )
        return True
    else:
        print(
            f"New model (silhouette={new_silhouette:.4f}) did NOT beat current Production "
            f"(silhouette={current_silhouette:.4f}) -> keeping version {current_prod.version} in Production."
        )
        return False
