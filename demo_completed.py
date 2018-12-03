# Train a mobilenet model using Tensorflow

from azureml.core import Experiment, Run, Workspace
from azureml.train.dnn import TensorFlow
from azureml.train.hyperdrive import (BanditPolicy, HyperDriveRunConfig, PrimaryMetricGoal, RandomParameterSampling, loguniform, uniform)
from azureml.train.widgets import RunDetails
from scripts.config import AML
from scripts.image_helpers import (get_sample_images_for_each_species, plot_images_in_grid)
from scripts.retrain import train

def inspect_data():
    # Examine the dataset
    get_ipython().run_line_magic('matplotlib', 'inline')
    images_data = get_sample_images_for_each_species('images')
    plot_images_in_grid(images_data, 6)


def retrieve_AML_config():
    # Get a reference to the Workspace and the Experiment
    ws = Workspace.get(name=AML.workspace_name, 
                       subscription_id=AML.subscription_id, 
                       resource_group=AML.resource_group) 
    experiment = Experiment(ws, AML.experiment_name)
    return ws, experiment

def transfer_learning():
    # Transfer learning using mobilenet and our pet images
    train(architecture='mobilenet_0.50_224', 
          image_dir='images', 
          output_dir='models', 
          bottleneck_dir='bottleneck',
          model_dir='model',
          learning_rate=0.00008, 
          use_hyperdrive=False)

def hyperparameter_tuning(ws,experiment):
    # Create and submit a Hyperdrive job
    cluster = ws.compute_targets[AML.compute_name]
    script_params={
        '--datastore-dir': ws.get_default_datastore().as_mount(),
    }
    tf_estimator = TensorFlow(source_directory='scripts',
                              compute_target=cluster,
                              entry_script='train.py',
                              script_params=script_params,
                              use_gpu=True)
    ps = RandomParameterSampling(
        {
            '--learning-rate': loguniform(-15, -3)
        }
    )
    early_termination_policy = BanditPolicy(slack_factor = 0.15, evaluation_interval=2)
    hyperdrive_run_config = HyperDriveRunConfig(estimator = tf_estimator, 
                                                hyperparameter_sampling = ps, 
                                                policy = early_termination_policy,
                                                primary_metric_name = "validation_accuracy",
                                                primary_metric_goal = PrimaryMetricGoal.MAXIMIZE,
                                                max_total_runs = 20,
                                                max_concurrent_runs = 4)

    hd_run = experiment.submit(hyperdrive_run_config)
    RunDetails(Run(experiment, hd_run.id)).show()
    return hd_run

def retrieve_best_model(hd_run):
    # Retrieve the best run from the Hyperdrive run
    best_run = hd_run.get_best_run_by_primary_metric()
    model = best_run.register_model(model_name='pet-detector', model_path='outputs')  

def train():
    inspect_data()
    transfer_learning()
    ws, experiment = retrieve_AML_config()
    hd_run = hyperparameter_tuning(ws, experiment)
    retrieve_best_model(hd_run)