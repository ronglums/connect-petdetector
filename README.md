# Transfer learning using mobilenet and the Oxford IIIT Pets Dataset

This repository contains a notebook that will teach you how to train a [mobilenet](https://arxiv.org/abs/1704.04861) model using the [Oxford IIIT Pets Dataset](http://www.robots.ox.ac.uk/~vgg/data/pets/). This dataset contains pictures of 37 different breeds of cats and dogs:

![Class Examples](images/class_examples.jpg)

Start this project in Azure Notebooks by clicking on this badge: [![Azure Notebooks](https://notebooks.azure.com/launch.svg)](https://notebooks.azure.com/import/gh/jflam/connect-demo)

# Setup

This demo has a number of steps that you'll need to complete to replicate what we did at the Connect() keynote.

## Azure Notebooks Setup

1. Sign up for an account on [Azure Notebooks](https://notebooks.azure.com) by logging in with either a Microsoft Account (e.g., an @outlook.com, an @live.com, or an @hotmail.com address) or an Azure Active Directory organization account (e.g., one from your workplace like @microsoft.com).
1. Open the [Github repository for the demo](https://github.com/Microsoft/connect-petdetector)
1. Scroll down until you see the Azure Notebooks Launch badge ![Azure Notebooks
   Badge](https://notebooks.azure.com/launch.svg) and click on it to clone it into your Azure Notebooks account
1. If you haven't already, sign up for a [Free Azure Subscription](https://azure.microsoft.com/en-us/free/?v=18.45) which will give you $200 in Azure Credits to spend on training your models using GPUs over the first 30 days of your scription.
1. Open the [Azure Portal](https://portal.azure.com) and create an Ubuntu-based [Data Science Virtual Machine](https://azure.microsoft.com/en-us/services/virtual-machines/data-science-virtual-machines/) (DSVM) using a GPU-powered Azure VM size. We used the [NC6](https://docs.microsoft.com/en-us/azure/virtual-machines/windows/sizes-gpu#nc-series) VM size in the Connect() demo, which is powered by an [NVidia K80 GPU](https://www.nvidia.com/en-gb/data-center/tesla-k80/).
1. Make sure that you set a username and password (not a SSH key) to log into your DSVM. You will use those same credentials when connecting to your VM from Azure Notebooks.
1. Switch back to the Azure Notebooks Project that you created. Click on the drop-down to select the DSVM that you created earlier, and click on the Run button to start it. In the dialog that appears, enter the username and password that you assigned when you created your DSVM.
1. This will open the Jupyter treeview of your project. Double-click on the
   `setup.ipynb` file to launch the setup notebook.
1. Follow the instructions in the notebook. The configuration cell contains a
   number of parameters that you will need to set:
```python
class AMLConfig:
    pass

AMLConfig.workspace_name           = ''
AMLConfig.experiment_name          = ''
AMLConfig.resource_group           = ''
AMLConfig.compute_name             = ''
AMLConfig.training_script_filename = 'train.py'
AMLConfig.scoring_script_filename  = 'score.py'
AMLConfig.subscription_id          = ''
AMLConfig.storage_account_name     = ''
AMLConfig.storage_account_key      = ''
AMLConfig.datastore_name           = 'default'
AMLConfig.container_name           = 'default'
AMLConfig.images_dir               = 'images'

AML = AMLConfig()
```
10. The `setup` notebook will download the dataset, transform the file
   layout into a form that Tensorflow can understand, and generate a number of
   intermediate _bottleneck_ files that contain pre-computed weights that will
   be used later when we apply transfer learning to retrain only the final layer
    in the model using the labelled images in the pet dataset. It also uploads
    the images into an AML Datastore that is used by the AML Compute cluster
    that is created at the end of the setup script.

# Visual Studio Code Setup

1. Download and install [Visual Studio
   Code](https://code.visualstudio.com/download)
1. Run Visual Studio Code and click on the Extensions tab. Search for and
   install the following extensions: `Python`, `Azure Machine Learning`,
   `Intellicode`
1. Install the latest [Anaconda Python
   distribution](https://www.anaconda.com/download)
1. Open a Terminal Window and run the following three commands to create a new
   conda environment:

```
conda create -n py36 python=3.6 jupyter pillow matplotlib numpy tensorflow=1.11

conda activate py36

pip install --upgrade azureml-sdk
```
5. Clone the [pet detector Github
   repo](https://github.com/Microsoft/connect-petdetector) onto your machine.

```
git clone https://github.com/Microsoft/connect-petdetector
```

6. Navigate to the directory that contains your git repo and type `code .` to
   launch Visual Studio Code.
