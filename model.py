import torch

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PhysicallyInformedNN(torch.nn.Module):
    def __init__(self, df, input_size, hidden_layers, output_size):
        super(PhysicallyInformedNN, self).__init__()
        # Constants throughout the model
        self.r = 0.01

        # Initialize input and output tensors        
        self.S = torch.tensor(df[' [UNDERLYING_LAST]'].astype(float).values, requires_grad=True).float().unsqueeze(1).to(device)
        self.t = torch.tensor(df[' [DTE]'].astype(float).values, requires_grad=True).float().unsqueeze(1).to(device)
        self.sigma_call = torch.tensor(df[' [C_IV]'].astype(float).values, dtype=torch.float32).float().unsqueeze(1).to(device)
        self.sigma_put = torch.tensor(df[' [P_IV]'].astype(float).values, dtype=torch.float32).float().unsqueeze(1).to(device)
        self.call_price = torch.tensor(df[' [C_LAST]'].astype(float).values, dtype=torch.float32).float().unsqueeze(1).to(device)
        self.put_price = torch.tensor(df[' [P_LAST]'].astype(float).values, dtype=torch.float32).float().unsqueeze(1).to(device)

        # Define the activation function and mse
        self.activation = torch.nn.Tanh()
        self.mse_loss = torch.nn.MSELoss()

        # Define the layers of the network
        layers = []
        layers.append(torch.nn.Linear(input_size, hidden_layers[0]))
        layers.append(self.activation)

        for i in range(len(hidden_layers) - 1):
            layers.append(torch.nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(self.activation)

        layers.append(torch.nn.Linear(hidden_layers[-1], output_size))
        
        self.network = torch.nn.Sequential(*layers)

        # Define the optimizer
        self.optimizer = torch.optim.Adam([{'params': self.parameters()}], lr=1e-3, betas=(0.9, 0.999), eps=1e-8)         

    def forward(self, x):
        return self.network(x)

    def loss(self):
        """
        Calculates separate physics-informed losses for call and put prices.
        Uses separate sigmas for call and put options.
        """
        inputs = torch.cat([self.S, self.t, self.sigma_call, self.sigma_put], dim=1)
        output = self.forward(inputs)
        call_price = output[:, 0:1]
        put_price = output[:, 1:2]

        # Partial derivatives for call price
        dC_dt = torch.autograd.grad(call_price, self.t, grad_outputs=torch.ones_like(call_price), retain_graph=True, create_graph=True)[0]
        dC_dS = torch.autograd.grad(call_price, self.S, grad_outputs=torch.ones_like(call_price), retain_graph=True, create_graph=True)[0]
        d2C_dS2 = torch.autograd.grad(dC_dS, self.S, grad_outputs=torch.ones_like(dC_dS), retain_graph=True, create_graph=True)[0]

        # Partial derivatives for put price
        dP_dt = torch.autograd.grad(put_price, self.t, grad_outputs=torch.ones_like(put_price), retain_graph=True, create_graph=True)[0]
        dP_dS = torch.autograd.grad(put_price, self.S, grad_outputs=torch.ones_like(put_price), retain_graph=True, create_graph=True)[0]
        d2P_dS2 = torch.autograd.grad(dP_dS, self.S, grad_outputs=torch.ones_like(dP_dS), retain_graph=True, create_graph=True)[0]

        # Black-Scholes residuals for call and put options
        bs_residual_call = dC_dt + 0.5 * self.sigma_call**2 * self.S**2 * d2C_dS2 + self.r * self.S * dC_dS - self.r * call_price
        bs_residual_put = dP_dt + 0.5 * self.sigma_put**2 * self.S**2 * d2P_dS2 + self.r * self.S * dP_dS - self.r * put_price

        # Mean squared physics-informed losses for call and put options
        physics_loss_call = torch.mean(bs_residual_call ** 2)
        physics_loss_put = torch.mean(bs_residual_put ** 2)

        data_call_loss = self.mse_loss(call_price, self.call_price)
        data_put_loss = self.mse_loss(put_price, self.put_price)

        total_loss = (data_call_loss + physics_loss_call) + (physics_loss_put + data_put_loss)

        return total_loss
    
    def train(self, epochs):
        """
        Trains the PhysicallyInformedNN model.

        Parameters:
        - model: The PhysicallyInformedNN model to train.
        - df: The DataFrame containing the training data.
        - input_size: The number of input features.
        - epochs: Number of training epochs.
        - r: Risk-free interest rate for loss calculation.
        """

        # Training loop
        for epoch in range(epochs):
            # Zero the gradients
            self.optimizer.zero_grad()

            # Forward pass to calculate loss
            loss = self.loss()

            # Backward pass to compute gradients
            loss.backward()

            # Update parameters
            self.optimizer.step()

            # Print loss every 100 epochs
            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}")

        print("Training complete.")