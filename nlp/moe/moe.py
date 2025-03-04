import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.fc(x)

class MoELayer(nn.Module):
    def __init__(self, num_experts, in_features, out_features):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([Linear(in_features, out_features) for _ in range(num_experts)])
        self.gate = Linear(in_features, num_experts)

    def forward(self, x):
        # x: [N, seq_len, in_features]
        # gate_score: [N, seq_len, num_experts]
        gate_score = F.softmax(self.gate(x), dim=-1)
        # expert_outputs: [N, seq_len, num_experts, out_features]
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)
        # gate_score: [N, seq_len, num_experts] -> [N, seq_len, 1, num_experts]
        # expert_outputs: [N, seq_len, num_experts, out_features]
        output = torch.einsum('nle,nleo->nlo', gate_score, expert_outputs)
        return output

batch_size = 2
seq_len = 3
input_size = 5

num_experts = 4
output_size = 3


model = MoELayer(num_experts, input_size, output_size)

demo = torch.randn(batch_size, seq_len, input_size)

output = model(demo)

print(output.shape)