#Example: https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb#scrollTo=kyrGCU6Hb-fu 
import pprint

class HyperParameterSearch:
    def __init__(self, method = 'random', metric='val_loss'):

        self.sweep_config = {
            'method': method
            }

        self.metric = {
            'name': metric,
            'goal': 'minimize'   
            }
        self.sweep_config['metric'] = self.metric


    def set_parameters_to_optimize_grid(self, batch_sizes, lrs_inner_model, lrs_outer_model):

        parameters_dict = {
            'batch_size': {
            'values': batch_sizes   #[2, 4, 8, 32, 64, 128]
            },
            'lr_inner_model': {
            'values': lrs_inner_model   #[0.001, 0.01, 0.003]
            },
            'lr_outer_model': {
            'values': lrs_outer_model 
            }
            # 'optimizer': {
            #     'values': ['adam', 'sgd']
            #     },
            # 'dropout': {
            #     'values': [0.15, 0.2, 0.25, 0.3, 0.4]
            #     },
            }

        self.sweep_config['parameters'] = parameters_dict
        pprint.pprint(self.sweep_config)
        
        return self.sweep_config


    def set_parameters_to_optimize_random(self, batch_sizes_min_max, lrs_inner_model_min_max, lrs_outer_model_min_max):
        parameters_dict = {
            'batch_size': {
                # integers between 32 and 256
                # with evenly-distributed logarithms 
                'distribution': 'q_log_uniform_values',
                'q': 8,
                'min': batch_sizes_min_max[0],
                'max': batch_sizes_min_max[1],
            },
            'lr_inner_model': {
                # a flat distribution between 0 and 0.1
                'distribution': 'uniform',
                'min': lrs_inner_model_min_max[0],
                'max': lrs_inner_model_min_max[1]
            },
            'lr_outer_model': {
                # a flat distribution between 0 and 0.1
                'distribution': 'uniform',
                'min': lrs_outer_model_min_max[0],
                'max': lrs_outer_model_min_max[1]
            }
            }


        self.sweep_config['parameters'] = parameters_dict
        pprint.pprint(self.sweep_config)
    
        return self.sweep_config