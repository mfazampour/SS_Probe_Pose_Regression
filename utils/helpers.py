from collections import defaultdict

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()).to(device) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def calculate_average_loss(list_of_dicts):
    """
    Calculate the average value for each key in a list of dictionaries.

    Parameters:
        list_of_dicts (list): List of dictionaries containing losses.

    Returns:
        dict: A dictionary containing the average value for each key.
    """
    # Initialize defaultdict with float as the default factory function
    average_dict = defaultdict(float)

    # Count number of dictionaries to calculate the average later
    num_dicts = len(list_of_dicts)

    # Check if the list is empty
    if num_dicts == 0:
        return "The list of dictionaries is empty."

    # Iterate over each dictionary in the list
    for d in list_of_dicts:
        # Iterate over each key-value pair in the dictionary
        for key, value in d.items():
            average_dict[key] += value

    # Calculate the average
    for key in average_dict:
        average_dict[key] /= num_dicts

    # Convert defaultdict to regular dict if desired
    return dict(average_dict)