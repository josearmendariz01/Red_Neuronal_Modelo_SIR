import torch  # Importa PyTorch, una biblioteca de aprendizaje automático
import torch.nn as nn  # Importa el módulo de redes neuronales de PyTorch
from torch.utils.data import Dataset  # Importa la clase base para datasets de PyTorch

# Definición de la clase SIRDataset que hereda de Dataset
class SIRDataset(Dataset):
    def __init__(self, S, I, R, beta, gamma):
        """
        Inicializa el dataset con S, I, R, beta y gamma.
        """
        self.S = S  # Asigna S al atributo de instancia self.S
        self.I = I  # Asigna I al atributo de instancia self.I
        self.R = R  # Asigna R al atributo de instancia self.R
        self.beta = beta  # Asigna beta al atributo de instancia self.beta
        self.gamma = gamma  # Asigna gamma al atributo de instancia self.gamma

    def __len__(self):
        """
        Retorna la longitud del dataset.
        """
        return len(self.S)  # Retorna la longitud del atributo self.S

    def __getitem__(self, idx):
        """
        Retorna un solo elemento del dataset dado un índice.
        """
        # Combina S, I y R en un solo tensor y lo convierte a tipo float32
        SIR = torch.tensor(self.S[idx].tolist() + self.I[idx].tolist() + self.R[idx].tolist(), dtype=torch.float32)
        # Crea un tensor con beta y gamma como objetivo
        target = torch.tensor([self.beta[idx], self.gamma[idx]], dtype=torch.float32)
        return SIR, target  # Retorna el tensor de entrada y el objetivo

# Definición de la clase SIRNetwork que hereda de nn.Module
class SIRNetwork(nn.Module):
    def __init__(self, input_size):
        """
        Inicializa la red neuronal con capas densas.
        """
        super(SIRNetwork, self).__init__()  # Llama al constructor de la clase base
        self.fc1 = nn.Linear(input_size, 64)  # Capa totalmente conectada con 64 neuronas
        self.fc2 = nn.Linear(64, 128)  # Capa totalmente conectada con 128 neuronas
        self.fc3 = nn.Linear(128, 64)  # Capa totalmente conectada con 64 neuronas
        self.fc4 = nn.Linear(64, 2)  # Capa totalmente conectada con 2 neuronas de salida

    def forward(self, x):
        """
        Propagación hacia adelante de los datos a través de la red.
        """
        x = torch.relu(self.fc1(x))  # Aplica la función de activación ReLU a la primera capa
        x = torch.relu(self.fc2(x))  # Aplica la función de activación ReLU a la segunda capa
        x = torch.relu(self.fc3(x))  # Aplica la función de activación ReLU a la tercera capa
        x = self.fc4(x)  # Pasa los datos por la última capa sin activación
        return x  # Retorna el resultado final
