#Pyhton code for initiation of the ANN Class
#numpy module is used for numeric operations 
import numpy as np
from typing import List, Tuple, Union

#initiationg the ANN class

class  ANN:
    def __init__(self, layer_details: List[int] =[], act_func: List[str]= [])-> None:
        """
        The ANN object has following attributes
        1. Layer details (as a list)
        2. Activation functions of each layers
        3. Cache memory to save immidiate results
        4. Layer types (To be developed)
        """
        self.parameters = {}
        self.layer_details = layer_details
        self.cache = []
        self.act_func = act_func
        self.cost_funct = ''
        self.hyp = 0 # regularization hyper parameter
        self.cal_grad = {}
        self.init_nn(layer_details) #initializing random values to weights and biases of the neurons

        #creting a method to initialize weights in the neurons
        def init_nn(self, layer_details : List[int])-> None:
            num_layers = int(len(self.layer_details))
            


