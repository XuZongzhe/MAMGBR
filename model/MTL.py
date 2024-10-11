import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads, head_dim):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.fc_query = nn.Linear(input_dim, num_heads * head_dim)
        self.fc_key = nn.Linear(input_dim, num_heads * head_dim)
        self.fc_value = nn.Linear(input_dim, num_heads * head_dim)
        self.fc_out = nn.Linear(num_heads * head_dim, input_dim)

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_query(x)
        K = self.fc_key(x)
        V = self.fc_value(x)
        Q = Q.view(batch_size, self.num_heads, self.head_dim)
        K = K.view(batch_size, self.num_heads, self.head_dim)
        V = V.view(batch_size, self.num_heads, self.head_dim)
        scores = torch.einsum('bhd,bhd->bh', Q, K) / (self.head_dim ** 0.5)
        attention = torch.softmax(scores, dim=-1)
        out = torch.einsum('bh,bhd->bhd', attention, V)
        out = out.reshape(batch_size, -1)
        out = self.fc_out(out)
        return out


class ExpertNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, head_dim):
        super(ExpertNet, self).__init__()
        self.attention = MultiHeadAttention(input_dim, num_heads, head_dim)
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.attention(x)
        x = self.fc(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x


class GateNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GateNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


class MTLNetwork(nn.Module):
    def __init__(self, input_dim, expert_dim, num_shared_experts, num_task_experts, num_heads, head_dim):
        super(MTLNetwork, self).__init__()
        self.num_shared_experts = num_shared_experts
        self.num_task_experts = num_task_experts
        self.expert_dim = expert_dim

        self.shared_experts = nn.ModuleList([
            ExpertNet(input_dim, expert_dim, num_heads, head_dim) for _ in range(num_shared_experts)
        ])
        self.task1_experts = nn.ModuleList([
            ExpertNet(input_dim, expert_dim, num_heads, head_dim) for _ in range(num_task_experts)
        ])
        self.task2_experts = nn.ModuleList([
            ExpertNet(input_dim, expert_dim, num_heads, head_dim) for _ in range(num_task_experts)
        ])

        gate_input_dim = input_dim
        self.gate_shared = GateNet(gate_input_dim, expert_dim, num_shared_experts)
        self.gate_task1 = GateNet(gate_input_dim, expert_dim, num_task_experts + num_shared_experts)
        self.gate_task2 = GateNet(gate_input_dim, expert_dim, num_task_experts + num_shared_experts)

    def forward(self, x):
        shared_expert_outputs = [expert(x) for expert in self.shared_experts]
        shared_expert_outputs = torch.stack(shared_expert_outputs,
                                            dim=1)  # [batch_size, num_shared_experts, expert_dim]

        task1_expert_outputs = [expert(x) for expert in self.task1_experts]
        task1_expert_outputs = torch.stack(task1_expert_outputs, dim=1)  # [batch_size, num_task_experts, expert_dim]

        task2_expert_outputs = [expert(x) for expert in self.task2_experts]
        task2_expert_outputs = torch.stack(task2_expert_outputs, dim=1)  # [batch_size, num_task_experts, expert_dim]

        shared_gate = self.gate_shared(x)  # [batch_size, num_shared_experts]
        task1_gate = self.gate_task1(x)  # [batch_size, num_task_experts + num_shared_experts]
        task2_gate = self.gate_task2(x)  # [batch_size, num_task_experts + num_shared_experts]

        task1_all_expert_outputs = torch.cat([shared_expert_outputs, task1_expert_outputs], dim=1)
        task2_all_expert_outputs = torch.cat([shared_expert_outputs, task2_expert_outputs], dim=1)

        shared_output = torch.einsum('bn,bnd->bd', shared_gate, shared_expert_outputs)
        task1_output = torch.einsum('bn,bnd->bd', task1_gate, task1_all_expert_outputs)
        task2_output = torch.einsum('bn,bnd->bd', task2_gate, task2_all_expert_outputs)

        return task1_output, task2_output


class Tower(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(Tower, self).__init__()
        layers = []
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.Dropout(0.2))
            input_dim = h_dim
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.network(x))


class MTL_Model(nn.Module):
    def __init__(self, num_feature, expert_dim, num_shared_experts, num_task_experts, num_heads, head_dim,
                 tower_hidden_dims):
        super(MTL_Model, self).__init__()

        self.mtl = MTLNetwork(
            input_dim=num_feature,
            expert_dim=expert_dim,
            num_shared_experts=num_shared_experts,
            num_task_experts=num_task_experts,
            num_heads=num_heads,
            head_dim=head_dim
        )

        self.task1_tower = Tower(expert_dim, tower_hidden_dims)
        self.task2_tower = Tower(expert_dim, tower_hidden_dims)

    def forward(self, x):
        task1_output, task2_output = self.ple(x)
        task1_output = self.task1_tower(task1_output)
        task2_output = self.task2_tower(task2_output)
        final_output = [task1_output, task2_output]
        return final_output
