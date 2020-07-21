import torch
import torch.nn as nn

input_dim = 3

class My_Net_V2(nn.Module):
    """Encoder is part of both TrajectoryGenerator and
    TrajectoryDiscriminator"""
    def __init__(
        self, seq_len=8, embedding_dim=30, first_h_dim=30, second_h_dim=60, mlp_dim=1024, num_layers=1,
        dropout=0.0, use_cuda=0
    ):
        super(My_Net_V2, self).__init__()

        self.mlp_dim = mlp_dim
        self.embedding_dim = embedding_dim
        self.first_h_dim = first_h_dim
        self.second_h_dim = second_h_dim
        self.num_layers = num_layers
        self.use_cuda = use_cuda
        self.seq_len = seq_len

        self.encoder = nn.LSTM(
            embedding_dim, first_h_dim, num_layers, dropout=dropout
        )

        self.state_embedding = nn.Linear(first_h_dim, second_h_dim)

        self.encoder2 = nn.LSTM(
            embedding_dim, second_h_dim, num_layers, dropout=dropout
        )

        self.hidden2pos = nn.Linear(second_h_dim, input_dim)

        self.relu = nn.ReLU()

        self.spatial_embedding = nn.Linear(input_dim, embedding_dim)

    def init_hidden(self, batch):
        state0 = torch.zeros(self.num_layers, batch, self.first_h_dim)
        state1 = torch.zeros(self.num_layers, batch, self.first_h_dim)

        if self.use_cuda == 1:
            state0 = state0.cuda()
            state1 = state1.cuda()

        return (state0, state1)

    def forward(self, obs_traj):

        batch = obs_traj.size(1)
        obs_traj_embedding = self.spatial_embedding(obs_traj.contiguous().view(-1, input_dim))
        obs_traj_embedding = obs_traj_embedding.view(-1, batch, self.embedding_dim)
        encoder_state_tuple = self.init_hidden(batch)
        output, state = self.encoder(obs_traj_embedding, encoder_state_tuple)

        state_0 = self.state_embedding(state[0])
        state_1 = self.state_embedding(state[1])

        output, state = self.encoder2(output, (state_0, state_1))

        cur_pos = self.hidden2pos(output.view(-1, self.second_h_dim))

        cur_pos = cur_pos.view(-1, batch, input_dim)

        return cur_pos
