import os
import azureml
from azureml.core import Experiment, Environment, Workspace, Run, ScriptRunConfig
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

ws = Workspace.from_config()
experiment = Experiment(Workspace=ws,name='network-training')
network_environment = Environment.from_conda_specification(name='training-env', file_path='train-environment.yml')


config = ScriptRunConfig(
    source_directory='./src',
    script='network.py',
    compute_target='network-dev',
    environment=network_environment
    )