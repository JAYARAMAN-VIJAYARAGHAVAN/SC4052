"""Shared visual CNN, 8-qubit PPO actor, classical critic and enemy detector."""
import pennylane as qml
import torch
from torch import nn
from .. import config as C

class QuantumCombatActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(C.FRAME_STACK, 16, 8, stride=4), nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3), nn.ReLU(), nn.Flatten(),
            nn.LazyLinear(128), nn.ReLU())
        self.to_angles = nn.Linear(128, C.N_QUBITS)
        self.weights = nn.Parameter(.05 * torch.randn(C.N_LAYERS, C.N_QUBITS, 3))
        self.policy = nn.Linear(C.N_QUBITS, C.N_ACTIONS)
        self.value = nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1))
        self.enemy_detector = nn.Linear(128, 1)
        device = qml.device('default.qubit', wires=C.N_QUBITS)
        @qml.qnode(device, interface='torch', diff_method='backprop')
        def circuit(inputs, weights):
            for layer in range(C.N_LAYERS):
                for wire in range(C.N_QUBITS):
                    qml.RY(inputs[wire], wires=wire)
                    qml.Rot(*weights[layer, wire], wires=wire)
                for wire in range(C.N_QUBITS):
                    qml.CNOT(wires=[wire, (wire + 1) % C.N_QUBITS])
            return [qml.expval(qml.PauliZ(w)) for w in range(C.N_QUBITS)]
        self.circuit = circuit

    def forward(self, images):
        if images.ndim == 3: images = images.unsqueeze(0)
        visual = self.encoder(images.float())
        angles = torch.pi * torch.tanh(self.to_angles(visual))
        quantum = torch.stack([
            torch.stack(self.circuit(row, self.weights)).to(visual.dtype)
            for row in angles])
        return (self.policy(quantum), self.value(visual).squeeze(-1),
                self.enemy_detector(visual).squeeze(-1))
