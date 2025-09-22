import torch

class MLP:
    def __init__(
        self,
        linear_1_in_features,
        linear_1_out_features,
        f_function,
        linear_2_in_features,
        linear_2_out_features,
        g_function
    ):
        """
        Args:
            linear_1_in_features: the in features of first linear layer
            linear_1_out_features: the out features of first linear layer
            linear_2_in_features: the in features of second linear layer
            linear_2_out_features: the out features of second linear layer
            f_function: string for the f function: relu | sigmoid | identity
            g_function: string for the g function: relu | sigmoid | identity
        """
        self.f_function = f_function
        self.g_function = g_function

        self.parameters = dict(
            W1 = torch.randn(linear_1_out_features, linear_1_in_features),
            b1 = torch.randn(linear_1_out_features),
            W2 = torch.randn(linear_2_out_features, linear_2_in_features),
            b2 = torch.randn(linear_2_out_features),
        )
        self.grads = dict(
            dJdW1 = torch.zeros(linear_1_out_features, linear_1_in_features),
            dJdb1 = torch.zeros(linear_1_out_features),
            dJdW2 = torch.zeros(linear_2_out_features, linear_2_in_features),
            dJdb2 = torch.zeros(linear_2_out_features),
        )

        # put all the cache value you need in self.cache
        self.cache = dict()

    def forward(self, x):
        """
        Args:
            x: tensor shape (batch_size, linear_1_in_features)
        """
        # First linear layer: z1 = W1 * x + b1
        z1 = torch.matmul(x, self.parameters['W1'].T) + self.parameters['b1']
        
        # Apply f function (activation)
        if self.f_function == 'relu':
            a1 = torch.relu(z1)
        elif self.f_function == 'sigmoid':
            a1 = torch.sigmoid(z1)
        elif self.f_function == 'identity':
            a1 = z1
        else:
            raise ValueError(f"Unknown f_function: {self.f_function}")
        
        # Second linear layer: z2 = W2 * a1 + b2
        z2 = torch.matmul(a1, self.parameters['W2'].T) + self.parameters['b2']
        
        # Apply g function (activation)
        if self.g_function == 'relu':
            y_hat = torch.relu(z2)
        elif self.g_function == 'sigmoid':
            y_hat = torch.sigmoid(z2)
        elif self.g_function == 'identity':
            y_hat = z2
        else:
            raise ValueError(f"Unknown g_function: {self.g_function}")
        
        # Store intermediate values for backward pass
        self.cache['x'] = x
        self.cache['z1'] = z1
        self.cache['a1'] = a1
        self.cache['z2'] = z2
        
        return y_hat
    
    def backward(self, dJdy_hat):
        """
        Args:
            dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
        """
        # Retrieve cached values
        x = self.cache['x']
        z1 = self.cache['z1']
        a1 = self.cache['a1']
        z2 = self.cache['z2']
        
        # Gradient w.r.t. z2 (before g function)
        if self.g_function == 'relu':
            # d/dz2 ReLU(z2) = 1 if z2 > 0, else 0
            dg_dz2 = (z2 > 0).float()
        elif self.g_function == 'sigmoid':
            # d/dz2 sigmoid(z2) = sigmoid(z2) * (1 - sigmoid(z2))
            sigmoid_z2 = torch.sigmoid(z2)
            dg_dz2 = sigmoid_z2 * (1 - sigmoid_z2)
        elif self.g_function == 'identity':
            # d/dz2 identity(z2) = 1
            dg_dz2 = torch.ones_like(z2)
        else:
            raise ValueError(f"Unknown g_function: {self.g_function}")
        
        dJdz2 = dJdy_hat * dg_dz2
        
        # Gradients for second layer (W2, b2)
        self.grads['dJdW2'] = torch.matmul(dJdz2.T, a1)  # (out_features, in_features)
        self.grads['dJdb2'] = torch.sum(dJdz2, dim=0)    # (out_features,)
        
        # Gradient w.r.t. a1 (output of first layer)
        dJda1 = torch.matmul(dJdz2, self.parameters['W2'])  # (batch_size, in_features)
        
        # Gradient w.r.t. z1 (before f function)
        if self.f_function == 'relu':
            # d/dz1 ReLU(z1) = 1 if z1 > 0, else 0
            df_dz1 = (z1 > 0).float()
        elif self.f_function == 'sigmoid':
            # d/dz1 sigmoid(z1) = sigmoid(z1) * (1 - sigmoid(z1))
            sigmoid_z1 = torch.sigmoid(z1)
            df_dz1 = sigmoid_z1 * (1 - sigmoid_z1)
        elif self.f_function == 'identity':
            # d/dz1 identity(z1) = 1
            df_dz1 = torch.ones_like(z1)
        else:
            raise ValueError(f"Unknown f_function: {self.f_function}")
        
        dJdz1 = dJda1 * df_dz1
        
        # Gradients for first layer (W1, b1)
        self.grads['dJdW1'] = torch.matmul(dJdz1.T, x)  # (out_features, in_features)
        self.grads['dJdb1'] = torch.sum(dJdz1, dim=0)   # (out_features,)

    
    def clear_grad_and_cache(self):
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()

def mse_loss(y, y_hat):
    """
    Args:
        y: the label tensor (batch_size, linear_2_out_features)
        y_hat: the prediction tensor (batch_size, linear_2_out_features)

    Return:
        J: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # MSE Loss: J = mean((y_hat - y)^2)
    loss = torch.mean((y_hat - y) ** 2)
    
    # Gradient of MSE loss w.r.t. y_hat: dJ/dy_hat = 2 * (y_hat - y) / (batch_size * output_dim)
    # Note: PyTorch's MSE loss is normalized by both batch_size and output_dim
    batch_size = y.shape[0]
    output_dim = y.shape[1]
    dJdy_hat = 2 * (y_hat - y) / (batch_size * output_dim)
    
    return loss, dJdy_hat

def bce_loss(y, y_hat):
    """
    Args:
        y_hat: the prediction tensor
        y: the label tensor
        
    Return:
        loss: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # BCE Loss: J = -mean(y * log(y_hat) + (1-y) * log(1-y_hat))
    # Add small epsilon to avoid log(0)
    epsilon = 1e-15
    y_hat_clipped = torch.clamp(y_hat, epsilon, 1 - epsilon)
    
    # BCE loss
    loss = -torch.mean(y * torch.log(y_hat_clipped) + (1 - y) * torch.log(1 - y_hat_clipped))
    
    # Gradient of BCE loss w.r.t. y_hat: dJ/dy_hat = (y_hat - y) / (y_hat * (1 - y_hat)) / (batch_size * output_dim)
    # Note: PyTorch's BCE loss is normalized by both batch_size and output_dim
    batch_size = y.shape[0]
    output_dim = y.shape[1]
    dJdy_hat = (y_hat_clipped - y) / (y_hat_clipped * (1 - y_hat_clipped)) / (batch_size * output_dim)
    
    return loss, dJdy_hat











