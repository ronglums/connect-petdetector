
class AMLConfig:
    pass

AMLConfig.workspace_name           = 'Connect'
AMLConfig.experiment_name          = 'ConnectExperiment'
AMLConfig.resource_group           = 'juliademo'
AMLConfig.compute_name             = 'nc6cluster'
AMLConfig.training_script_filename = 'train.py'
AMLConfig.scoring_script_filename  = 'score.py'
AMLConfig.subscription_id          = '15ae9cb6-95c1-483d-a0e3-b1a1a3b06324'
AMLConfig.storage_account_name     = 'connectstoragephbotvhz'
AMLConfig.storage_account_key      = '7lf2UqasSb7MAi0n3Hf34VBeYr1IJ/EEUOE4r7MWK41tVmaq8cfjIXgJVX+A41y/bV5QyNFpYrYhkzK8JPOpNA=='
AMLConfig.datastore_name           = 'default'
AMLConfig.container_name           = 'default'
AMLConfig.images_dir               = 'images'

AML = AMLConfig()
