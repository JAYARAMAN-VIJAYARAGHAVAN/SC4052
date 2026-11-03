"""Eight-qubit high-level actor; classical value function."""
import pennylane as qml
import torch
from torch import nn
from .. import config as C

class QuantumActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv2d(C.FRAME_STACK,16,8,stride=4),nn.ReLU(),
            nn.Conv2d(16,32,4,stride=2),nn.ReLU(),nn.Conv2d(32,32,3),nn.ReLU(),
            nn.Flatten(),nn.LazyLinear(128),nn.ReLU())
        self.fusion=nn.Sequential(nn.Linear(128+C.N_NAV_FEATURES,64),nn.Tanh())
        self.to_angles=nn.Linear(64,C.N_QUBITS)
        self.weights=nn.Parameter(.05*torch.randn(C.N_LAYERS,C.N_QUBITS,3))
        self.policy=nn.Linear(C.N_QUBITS,C.N_OPTIONS)
        self.value=nn.Sequential(nn.Linear(128+C.N_NAV_FEATURES,64),nn.Tanh(),nn.Linear(64,1))
        device=qml.device('default.qubit',wires=C.N_QUBITS)
        @qml.qnode(device,interface='torch',diff_method='backprop')
        def circuit(inputs,weights):
            for layer in range(C.N_LAYERS):
                for wire in range(C.N_QUBITS):
                    qml.RY(inputs[wire],wires=wire)
                    qml.Rot(*weights[layer,wire],wires=wire)
                for wire in range(C.N_QUBITS):
                    qml.CNOT(wires=[wire,(wire+1)%C.N_QUBITS])
            return [qml.expval(qml.PauliZ(w)) for w in range(C.N_QUBITS)]
        self.circuit=circuit
    def forward(self,obs):
        image=obs['image'].float();nav=obs['nav'].float();mask=obs['mask'].bool()
        if image.ndim==3:image=image.unsqueeze(0);nav=nav.unsqueeze(0);mask=mask.unsqueeze(0)
        visual=self.encoder(image);merged=torch.cat([visual,nav],dim=-1)
        angles=torch.pi*torch.tanh(self.to_angles(self.fusion(merged)))
        q=torch.stack([torch.stack(self.circuit(row,self.weights)).to(visual.dtype) for row in angles])
        logits=self.policy(q).masked_fill(~mask,-1e9)
        return logits,self.value(merged).squeeze(-1)
