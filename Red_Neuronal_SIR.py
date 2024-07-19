import torch  # Importa PyTorch, una biblioteca de aprendizaje automático
import torch.nn as nn  # Importa el módulo de redes neuronales de PyTorch
from torch.utils.data import DataLoader, random_split  # Importa herramientas para manejar datasets
import numpy as np  # Importa NumPy para manejar arrays
from sklearn.preprocessing import RobustScaler  # Importa RobustScaler para normalizar datos
import matplotlib.pyplot as plt  # Importa Matplotlib para graficar

# Importación de las clases definidas en Imports.py
from Imports import SIRDataset
from Imports import SIRNetwork

# Definición del dispositivo (GPU si está disponible, sino CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cargar el dataset
DataSetSIR = np.load("DataSetSIR5.npz")

# Inicializar los escaladores robustos para normalización
s_scaler = RobustScaler()
i_scaler = RobustScaler()
r_scaler = RobustScaler()
beta_scaler = RobustScaler()
gamma_scaler = RobustScaler()

# Extraer datos del archivo cargado
S = DataSetSIR['S_']
I = DataSetSIR['I_']
R = DataSetSIR['R_']
gamma = DataSetSIR['gamma_']
Beta = DataSetSIR['Beta_']

# Ajustar y transformar los datos usando los escaladores
S = s_scaler.fit_transform(S)
I = i_scaler.fit_transform(I)
R = r_scaler.fit_transform(R)
Beta = beta_scaler.fit_transform(Beta.reshape(-1, 1)).squeeze()
gamma = gamma_scaler.fit_transform(gamma.reshape(-1, 1)).squeeze()

# Crear un objeto SIRDataset con los datos normalizados
sir_data = SIRDataset(S, I, R, Beta, gamma)

# Determinar las longitudes para dividir el dataset en entrenamiento y validación
total_len = len(sir_data)
train_len = int(0.8 * total_len)
val_len = total_len - train_len

# Dividir aleatoriamente el dataset en conjunto de entrenamiento y validación
train_dataset, val_dataset = random_split(sir_data, [train_len, val_len])

# Crear DataLoaders para el entrenamiento y la validación
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Definir el tamaño de entrada del modelo
n = len(S[0])  # Asumiendo que S, I y R tienen la misma longitud

# Inicializar el modelo y moverlo al dispositivo (GPU o CPU)
SIRmodel = SIRNetwork(input_size=3*n).to(device)

# Inicializar listas para almacenar los valores de las métricas
train_losses = []
val_losses = []

# Definir la función de pérdida y el optimizador
loss_function = nn.HuberLoss(delta=0.5)
nTrainSteps = 150
optimizer = torch.optim.Adam(SIRmodel.parameters(), lr=1e-4)

# Entrenamiento del modelo
for epoch in range(0, nTrainSteps):
    SIRmodel.train()  # Establecer el modo de entrenamiento
    train_loss = 0.0  # Inicializar la pérdida de entrenamiento
    for i, data in enumerate(train_loader, 0):
        inputs, targets = data[0].to(device), data[1].to(device)  # Mover datos al dispositivo
        optimizer.zero_grad()  # Poner a cero los gradientes
        outputs = SIRmodel(inputs)  # Propagación hacia adelante
        loss = loss_function(outputs, targets)  # Calcular la pérdida
        loss.backward()  # Propagación hacia atrás
        optimizer.step()  # Actualizar los parámetros del modelo
        train_loss += loss.item()  # Acumular la pérdida
    train_losses.append(train_loss / len(train_loader))  # Guardar la pérdida de entrenamiento

    SIRmodel.eval()  # Establecer el modo de evaluación
    val_loss = 0.0  # Inicializar la pérdida de validación
    with torch.no_grad():  # Desactivar el cálculo de gradientes
        for batch in val_loader:
            x, y = batch[0].to(device), batch[1].to(device)  # Mover datos al dispositivo
            predictions = SIRmodel(x)  # Propagación hacia adelante
            loss = loss_function(predictions, y)  # Calcular la pérdida
            val_loss += loss.item()  # Acumular la pérdida
    val_losses.append(val_loss / len(val_loader))  # Guardar la pérdida de validación

    if (epoch + 1) % 15 == 0:  # Imprimir cada 15 épocas
        print(f"Epoch {epoch+1}, Training Loss: {train_loss/len(train_loader)}, Validation Loss: {val_loss/len(val_loader)}")

# Graficar la pérdida de entrenamiento y validación
plt.plot(range(1, nTrainSteps + 1), train_losses, label='Training Loss')
plt.plot(range(1, nTrainSteps + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.show()

# Guardar los parámetros del modelo entrenado
torch.save(SIRmodel.state_dict(), 'sir_model_params_Series_Especial_1.pth')
