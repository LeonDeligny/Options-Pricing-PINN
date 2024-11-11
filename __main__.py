import torch

from load import load_data
from model import PhysicallyInformedNN

if __name__ == "__main__":
	# Load data in dataframe
	df = load_data()

	# Extract columns if DataFrame and columns are valid
	S = torch.tensor(df[' [UNDERLYING_LAST]'].values, dtype=torch.float32, requires_grad=True)
	t = torch.tensor(df[' [DTE]'].values, dtype=torch.float32, requires_grad=True)
	sigma_call = torch.tensor(df[' [C_IV]'].values, dtype=torch.float32)
	sigma_put = torch.tensor(df[' [P_IV]'].values, dtype=torch.float32)
	call_price = torch.tensor(df[' [C_LAST]'].values, dtype=torch.float32)
	put_price = torch.tensor(df[' [P_LAST]'].values, dtype=torch.float32)

	# Create input and output tensors for training
	inputs = torch.stack((S, t, sigma_call, sigma_put), dim=1)
	targets = torch.stack((call_price, put_price), dim=1)  # Stack call and put prices for output

	# Set hyperparameters
	input_size = 4  # S and t as inputs
	hidden_layers = [64, 64, 64]
	output_size = 2  # Predicting call and put prices
	learning_rate = 0.001
	epochs = 1000
	batch_size = 64

	# Initialize the model, loss function, and optimizer
	model = PhysicallyInformedNN(df, input_size, hidden_layers, output_size)
	model.train(epochs=10_000)