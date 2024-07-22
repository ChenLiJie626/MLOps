from networkx import enumerate_all_cliques
import numpy as np
from fastapi import FastAPI
from fastapi import BackgroundTasks
from urllib.parse import urlparse
import mlflow
from mlflow.tracking import MlflowClient
from backend.models import DeleteApiData, TrainApiData, PredictApiData, User
from secretflow.ml.nn import FLModel
from .utils import initialize_secretflow, create_devices, create_dataset_builderVNet
from ml.networks.vnet import VNet
import torch
from .aggregator_1mask_hhash import SecureAggregator
import threading
import sys
import io
import time

class CustomWriter(io.TextIOBase):
    def __init__(self):
        self.content = ""
        self._lock = threading.Lock()

    def write(self, text):
        with self._lock:
            self.content += text

    def read_and_clear(self):
        with self._lock:
            # 只返回非空行
            lines = self.content.splitlines()
            self.content = ""
        return "\n".join(line for line in lines if line.strip())

    def flush(self):
        pass

def run_fit(custom_writer, model, data, data_builder_dict):
    original_stdout = sys.stdout
    sys.stdout = custom_writer

    try:
        model.fit(
            data,
            None,
            validation_data=data,
            epochs=1,
            aggregate_freq=1,
            dataset_builder=data_builder_dict,
        )
    finally:
        # 还原标准输出
        sys.stdout = original_stdout
output_file = "output.txt"
def print_output_periodically(custom_writer, fit_thread):
    with open(output_file, 'w') as f:
        while fit_thread.is_alive():
            time.sleep(5)  # 每隔5秒写入一次
            output = custom_writer.read_and_clear()
            if output:
                f.write(output + '\n')
                f.flush()
        # 写入剩余的内容
        output = custom_writer.read_and_clear()
        if output:
            f.write(output + '\n')

    
#mlflow.set_tracking_uri('sqlite:///backend.db')
mlflow.set_tracking_uri("sqlite:///db/backend.db")
app = FastAPI()
mlflowclient = MlflowClient(
    mlflow.get_tracking_uri(), mlflow.get_registry_uri())

model_path = 'models/VNet.pth'
model_name = 'MedSeg'

'''maintain a user list'''
users_list = []

def train_model_task(fl_model, data, data_builder_dict):
    """Tasks that trains the model. This is supposed to be running in the background
    Since it's a heavy computation it's better to use a stronger task runner like Celery
    For the simplicity I kept it as a fastapi background task"""

    # Set MLflow tracking
    mlflow.set_experiment("LITS17")
    with mlflow.start_run() as run:
        # Log hyperparameters
        # mlflow.log_params(hyperparams)

        # Train
        print("Training model")
        
        custom_writer = CustomWriter()

        fit_thread = threading.Thread(target=run_fit, args=(custom_writer, fl_model, data, data_builder_dict))
        print_thread = threading.Thread(target=print_output_periodically, args=(custom_writer, fit_thread))

        fit_thread.start()
        print_thread.start()

        fit_thread.join()
        print_thread.join()

        fl_model.save_model(model_path)
        # print("Logging results")
        # # Log in mlflow
        # for metric_name, metric_values in history.items():
        #     for metric_value in metric_values:
        #         mlflow.log_metric(metric_name, metric_value)

        # Register model
        model = VNet()
        model.load_state_dict(torch.load(model_path)['model_state_dict'])

        # Register model
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        print(f"{tracking_url_type_store=}")

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            mlflow.pytorch.log_model(
                model, "LinearModel", registered_model_name=model_name, conda_env=mlflow.pytorch.get_default_conda_env())
        else:
            mlflow.pytorch.log_model(
                model, "LinearModel-MNIST", registered_model_name=model_name)
        # Transition to production. We search for the last model with the name and we stage it to production
        mv = mlflowclient.search_model_versions(
            f"name='{model_name}'")[-1]  # Take last model version
        mlflowclient.transition_model_version_stage(
            name=mv.name, version=mv.version, stage="production")


@app.get("/")
async def read_root():
    return {"Tracking URI": mlflow.get_tracking_uri(),
            "Registry URI": mlflow.get_registry_uri()}


@app.get("/models")
async def get_models_api():
    """Gets a list with model names"""
    model_list = mlflowclient.search_registered_models()
    model_list = [model.name for model in model_list]
    return model_list


@app.post("/train")
async def train_api( background_tasks: BackgroundTasks): # data: TrainApiData,
    """Creates a model based on hyperparameters and trains it."""
    initialize_secretflow()
    alice, bob, carol, dave, device_list, aggregator = create_devices()
    aggregator = SecureAggregator(device=dave, participants=[alice, bob, carol])
    fl_model = FLModel(
            server=dave,
            device_list=device_list,
            model=VNet,
            aggregator=aggregator,
            sparsity=0.0,
            strategy="fed_avg_w",
            backend="torch",
            random_seed=1234,
            num_gpus=2
        )
    #todo 这里设置需要考虑是否需要一个文件存储系统，目前就用本地路径  
    data = {
        alice: 'ml/LITS17',
        bob: 'ml/LITS17',
        carol: 'ml/LITS17'
    }
    data_builder_dict = {
        alice: create_dataset_builderVNet(batch_size=2, random_seed=1234),
        bob: create_dataset_builderVNet(batch_size=2, random_seed=1234),
        carol: create_dataset_builderVNet(batch_size=2, random_seed=1234)
    }

    background_tasks.add_task(
        train_model_task, fl_model, data, data_builder_dict)

    return {"result": "Training task started"}


@app.post("/predict")
async def predict_api(data: PredictApiData):
    """Predicts on the provided image"""
    img = data.input_image
    model_name = data.model_name
    # Fetch the last model in production
    model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{model_name}/Production"
    )
    # Preprocess the image
    # Flatten input, create a batch of one and normalize
    img = np.array(img, dtype=np.float32).flatten()[np.newaxis, ...] / 255
    # Postprocess result
    pred = model.predict(img)
    print(pred)
    res = int(np.argmax(pred[0]))
    return {"result": res}


@app.post("/delete")
async def delete_model_api(data: DeleteApiData):
    model_name = data.model_name
    version = data.model_version
    
    if version is None:
        # Delete all versions
        mlflowclient.delete_registered_model(name=model_name)
        response = {"result": f"Deleted all versions of model {model_name}"}
    elif isinstance(version, list):
        for v in version:
            mlflowclient.delete_model_version(name=model_name, version=v)
        response = {
            "result": f"Deleted versions {version} of model {model_name}"}
    else:
        mlflowclient.delete_model_version(name=model_name, version=version)
        response = {
            "result": f"Deleted version {version} of model {model_name}"}
    return response

'''add_user'''
@app.post("/update_user")
async def update_user_api(user: User):
    if user.server_address != "":
        users_list.append(user) # add to the user_list
    server_address = user.server_address
    name = user.name
    role = user.role
    gpu = user.gpu
    print(server_address, name, role, gpu)
    return {"users_list": users_list}