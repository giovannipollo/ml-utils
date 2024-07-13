import torch.nn as nn
from .base_model import BaseModel


class SimpleNN_MNIST(BaseModel):
    """
    A simple neural network model with configurable layers and activation functions for MNIST
    """

    def __init__(self, config):
        """
        Simple Neural Network to train on

        Args:
            config (dict): Configuration dictionary containing model parameters.
        """
        super(SimpleNN_MNIST, self).__init__()
        model_config = config["model"]

        # Define layers
        self.fc1 = nn.Linear(model_config["input_size"], model_config["hidden_size"])
        self.activation = self.get_activation(model_config["activation"])
        self.dropout = nn.Dropout(model_config["dropout_rate"])
        self.fc2 = nn.Linear(model_config["hidden_size"], model_config["output_size"])

        # Weight Initialization
        if model_config["weight_init"] == "xavier":
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
        elif model_config["weight_init"] == "he":
            nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity="relu")
            nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity="relu")

    def get_activation(self, activation_name):
        """
        Get the activation function based on the provided name.

        Args:
            activation_name (str): Name of the activation function.

        Returns:
            nn.Module: The corresponding PyTorch activation function.

        Raises:
            ValueError: If an unsupported activation function is specified.
        """
        if activation_name == "relu":
            return nn.ReLU()
        elif activation_name == "sigmoid":
            return nn.Sigmoid()
        elif activation_name == "tanh":
            return nn.Tanh()
        else:
            raise ValueError("Unsupported activation function")

    def forward(self, x):
        """
        Defines the forward pass of the neural network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output of the neural network.
        """
        x = x.view(-1, 28 * 28)
        out = self.fc1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out
