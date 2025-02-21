import mlflow 
def connect_MLflow_lan():
    mlflow.set_tracking_uri("http://192.168.86.37:5000")

def connect_MLflow_local():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")