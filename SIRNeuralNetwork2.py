import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt

#Import Data Loaders
from Imports import SIRDataset

#Import Models
from Imports import SIRNetwork
from Imports import ImprovedSIRNetwork
from Imports import ResNetSIR

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Open the Data Set

DataSetSIR = np.load("DataSetSIR5.npz")

# Initialize the robust scalers
s_scaler = RobustScaler()
i_scaler = RobustScaler()
r_scaler = RobustScaler()
beta_scaler = RobustScaler()
gamma_scaler = RobustScaler()


S = DataSetSIR['S_']
I = DataSetSIR['I_']
R = DataSetSIR['R_']
gamma = DataSetSIR['gamma_']
Beta = DataSetSIR['Beta_']

# Assume S, I, R are NumPy arrays with shape (num_samples, time_steps)
# Fit and transform the data

S = s_scaler.fit_transform(S)
I = i_scaler.fit_transform(I)
R = r_scaler.fit_transform(R)
Beta = beta_scaler.fit_transform(Beta.reshape(-1, 1)).squeeze()
gamma = gamma_scaler.fit_transform(gamma.reshape(-1, 1)).squeeze()

# Create a SIRDataset object
sir_data = SIRDataset(S, I, R, Beta, gamma)

# Determine lengths
total_len = len(sir_data)
train_len = int(0.8 * total_len)
val_len = total_len - train_len

# Randomly split dataset into training set and validation set
train_dataset, val_dataset = random_split(sir_data, [train_len, val_len])

# Create DataLoaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Model dimension parameters
n = len(S[0])  # Assuming S, I, and R lists all have the same length

SIRmodel = SIRNetwork(input_size=3*n).to(device)

##################################################
## Train Model
##################################################

# Initialize lists to store the metric values:
train_losses = []
val_losses = []

# Define the loss function and optimizer
loss_function = nn.HuberLoss(delta=0.5)
nTrainSteps = 150
optimizer = torch.optim.Adam(SIRmodel.parameters(), lr=1e-4)

for epoch in range(0, nTrainSteps):

  # Set current loss value
    SIRmodel.train()
    train_loss = 0.0

  # Iterate over the DataLoader for training data
    for i, data in enumerate(train_loader, 0):
        # Get inputs
        inputs, targets = data[0].to(device), data[1].to(device)
        # Zero the gradients
        optimizer.zero_grad()
        # Perform forward pass (make sure to supply the input in the right way)
        outputs = SIRmodel(inputs)
        # Compute loss
        loss = loss_function(outputs, targets)
        # Perform backward pass
        loss.backward()
        # Perform optimization
        optimizer.step()
        # Print statistics
        train_loss += loss.item()

    train_losses.append(train_loss/len(train_loader))
    # Validation
    SIRmodel.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch[0].to(device),batch[1].to(device)
            predictions = SIRmodel(x)
            loss = loss_function(predictions, y)
            val_loss += loss.item()
    val_losses.append(val_loss/len(val_loader))

    if (epoch+1)%15 == 0:
        print(f"Epoch {epoch+1}, Training Loss: {train_loss/len(train_loader)}, Validation Loss: {val_loss/len(val_loader)}")

# Process is complete.
print('Training process has finished.')

plt.plot(range(1, nTrainSteps + 1), train_losses, label='Training Loss')
plt.plot(range(1, nTrainSteps + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.show()


#Save model parameters
torch.save(SIRmodel.state_dict(), 'sir_model_params_Series_Especial_1.pth')


