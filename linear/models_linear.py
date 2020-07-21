import torch
import torch.nn as nn

input_dim = 3


class LinearModel(nn.Module):
    """
    LinearModel.
    """
    def __init__(
        self, seq_len=8, h_dim=64,  use_cuda=0,embedding_dim=64
    ):
        super(LinearModel, self).__init__()
        self.h_dim = h_dim

        self.layer1 = torch.nn.Linear(input_dim, h_dim)
        self.layer2 = torch.nn.Linear(h_dim, input_dim)

        # self.mlp_dim = mlp_dim

        # self.embedding_dim = embedding_dim
        # self.num_layers = num_layers
        self.use_cuda = use_cuda
        # self.seq_len = seq_len


        # self.encoder = nn.LSTM(
        #     embedding_dim, h_dim, num_layers, dropout=dropout
        # )
        #
        # self.decoder = nn.LSTM(
        #     embedding_dim, h_dim, num_layers, dropout=dropout
        # )

        # self.hidden2pos = nn.Linear(h_dim, input_dim)
        #
        # self.spatial_embedding = nn.Linear(input_dim, embedding_dim)
    #
    # def init_hidden(self, batch):
    #     state0 = torch.zeros(self.num_layers, batch, self.h_dim)
    #     state1 = torch.zeros(self.num_layers, batch, self.h_dim)
    #
    #     if self.use_cuda == 1:
    #         state0 = state0.cuda()
    #         state1 = state1.cuda()
    #
    #     return (state0, state1)

    def forward(self, obs_traj):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 3)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """
        # Encode observed Trajectory
        result = []

        batch = obs_traj.size(1)
        obs_traj_embedding = self.layer1(obs_traj.contiguous().view(-1, input_dim))
        obs_traj_embedding = obs_traj_embedding.view(
            -1, batch, self.h_dim
         )
        # encoder_state_tuple = self.init_hidden(batch)
        # output, state = self.encoder(obs_traj_embedding, encoder_state_tuple)

        # decoder_c = torch.zeros(
        #     self.num_layers, batch, self.decoder_h_dim
        # )

        cur_pos = self.layer2(obs_traj_embedding.view(-1, self.h_dim))
        cur_pos = cur_pos.view(-1, batch, input_dim)

        return cur_pos


