import torch
import torch.nn as nn

input_dim = 3

class My_Net(nn.Module):
    """Encoder is part of both TrajectoryGenerator and
    TrajectoryDiscriminator"""
    def __init__(
        self, seq_len=8, embedding_dim=64, h_dim=64, mlp_dim=1024, num_layers=1,
        dropout=0.0, use_cuda=0
    ):
        super(My_Net, self).__init__()

        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.use_cuda = use_cuda
        self.seq_len = seq_len

        self.encoder = nn.LSTM(
            embedding_dim, h_dim, num_layers, dropout=dropout
        )

        self.decoder = nn.LSTM(
            embedding_dim, h_dim, num_layers, dropout=dropout
        )

        self.hidden2pos = nn.Linear(h_dim, input_dim)

        self.spatial_embedding = nn.Linear(input_dim, embedding_dim)

    def init_hidden(self, batch):
        state0 = torch.zeros(self.num_layers, batch, self.h_dim)
        state1 = torch.zeros(self.num_layers, batch, self.h_dim)

        if self.use_cuda == 1:
            state0 = state0.cuda()
            state1 = state1.cuda()

        return (state0, state1)

    def forward(self, obs_traj):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 3)
        Output:
        obs_traj shape is  torch.Size([8, 64, 3])
        obs_traj_embedding shape is  torch.Size([512, 64])
        # obs_traj_embedding shape is  torch.Size([512, 3])
        obs_traj_embedding shape is  torch.Size([8, 64, 64])
        ouput shape is  torch.Size([8, 64, 64])
        cur_pos shape is torch.Size([512, 3])
        cur_pos shape is torch.Size([8, 64, 3])
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """
        # Encode observed Trajectory
        result = []

        batch = obs_traj.size(1)
        # print("obs_traj shape is ",obs_traj.shape)
        # print("obs_traj shape is ",obs_traj.contiguous().view(-1, input_dim).shape)
        obs_traj_embedding = self.spatial_embedding(obs_traj.contiguous().view(-1, input_dim))
        # print("obs_traj_embedding shape is ",obs_traj_embedding.shape)
        obs_traj_embedding = obs_traj_embedding.view(
            -1, batch, self.embedding_dim
        )
        # print("obs_traj_embedding shape is ",obs_traj_embedding.shape)
        encoder_state_tuple = self.init_hidden(batch)
        output, state = self.encoder(obs_traj_embedding, encoder_state_tuple)

        # decoder_c = torch.zeros(
        #     self.num_layers, batch, self.decoder_h_dim
        # )
        # print("ouput shape is ",output.shape)

        cur_pos = self.hidden2pos(output.view(-1, self.h_dim))
        # print("cur_pos shape is {}".format(cur_pos.shape))
        cur_pos = cur_pos.view(-1, batch, input_dim)
        # print("cur_pos shape is {}".format(cur_pos.shape))

        return cur_pos
