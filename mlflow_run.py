import mlflow
from mlflow import sklearn
from model import get_model

##mlflow.set_experiment('Prediccion-contaminacion')
with mlflow.start_run():
    
    conda = {
        'name': 'mlflow-env',
        'channels': ['defaults'],
        'dependencies': [
            'python=3.7.0',
            'scikit-learn=0.19.2'
        ]
    }

    mlflow.sklearn.log_model(get_model(), 'model', conda_env = conda)
    artifact_path = mlflow.get_artifact_uri()
    artifact_path = artifact_path.replace("file://", "")
    print(artifact_path)

    f = open("/mlflow/mlflow_run.txt", "w")
    f.write("mlflow models serve -m "+artifact_path+"/model -h 0.0.0.0 -p 1234 -w 1")
    f.close()
    #os.environ["MLFLOW_ARTIFACT_PATH"] = artifact_path
