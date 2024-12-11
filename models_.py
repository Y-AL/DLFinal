# models
from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch
from transformers import ViTConfig, ViTModel

# schedulers
from enum import auto, Enum
import math

# normalizer
import torch

# dataset
from typing import NamedTuple, Optional
import torch
import numpy as np

from tqdm.auto import tqdm

def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)


class MockModel(torch.nn.Module):
    """
    Does nothing. Just for testing.
    """

    def __init__(self, device="cuda", bs=64, n_steps=17, output_dim=256):
        super().__init__()
        self.device = device
        self.bs = bs
        self.n_steps = n_steps
        self.repr_dim = 256

    def forward(self, states, actions):
        """
        Args:
            During training:
                states: [B, T, Ch, H, W]
            During inference:
                states: [B, 1, Ch, H, W]
            actions: [B, T-1, 2]

        Output:
            predictions: [B, T, D]
        """
        return torch.randn((self.bs, self.n_steps, self.repr_dim)).to(self.device)


class Encoder_ViT(nn.Module):
    def __init__(self, reprst_H, reprst_W):
        super().__init__()
        self.reprst_H = reprst_H
        self.reprst_W = reprst_W
         # Initializing a ViT vit-base-patch16-224 style configuration
        self.config_ViT = ViTConfig(
            hidden_size=self.reprst_H*self.reprst_W, # todo
            num_hidden_layers=4, 
            num_attention_heads=1, 
            intermediate_size=512, 
            image_size=65, 
            patch_size=13, 
            num_channels=2,
            return_dict=True
        )
        # Initializing a model (with random weights) from the vit-base-patch16-224 style configuration
        self.bareViT = ViTModel(self.config_ViT) 
        # The bare ViT Model transformer outputting raw hidden-states without any specific head on top. 
        """
        last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size))
        — Sequence of hidden-states at the output of the last layer of the model.
        
        pooler_output (torch.FloatTensor of shape (batch_size, hidden_size))
        — Last layer hidden-state of the first token of the sequence (classification token) after further processing through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns the classification token after processing through a linear layer and a tanh activation function. The linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.
        """

    def forward(self, observs):
        """
        Args:
            During training:
                observs: [B, Ch, H, W]
            During inference: will unroll the JEPA world model recurrently into the future, conditioned on "initial" observation and action sequence 
                observs: [B, Ch, H, W] ??
        Output:
            target_states: [B, reprst_H*reprst_W] or [B, reprst_H, reprst_W]
        """
        target_states = self.bareViT(observs)
        return target_states.pooler_output # [B, hidden_size] = [B, reprst_H*reprst_W]


class Predictor_1dCNN(nn.Module):
    def __init__(self, reprst_H, reprst_W):
        super().__init__()
        self.reprst_D = reprst_H*reprst_W
        self.action_projector = nn.Sequential(
            nn.Linear(2, self.reprst_D),
            nn.ReLU()
        )
        # input: [B, 2, reprst_H*reprst_W]
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=16, kernel_size=3, padding=1), # ch1: prev_states, ch2: actions_proj
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(in_channels=32, out_channels=1, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.BatchNorm2d(64),
            # nn.Conv2d(in_channels=64, out_channels=2, kernel_size=3, padding=1)
        )
        # input: [B, 1, reprst_H*reprst_W]

    def forward(self, prev_states, actions):
        """
        Args:
            During training:
                prev_states: [B, reprst_H*reprst_W]
            During inference: will unroll the JEPA world model recurrently into the future, conditioned on "initial" observation and action sequence 
                prev_states: [B, reprst_H*reprst_W]
            actions: [B, 2]
        Output:
            curr_states: [B, reprst_H*reprst_W]
        """
        actions_proj = self.action_projector(actions) # [B, reprst_H*reprst_W]
        input = torch.stack((prev_states, actions_proj), dim=1) # input: [B, 2, reprst_H*eprst_W]
        curr_states = self.cnn(input) # [B, 1, reprst_H*reprst_W]
        curr_states = curr_states.view(-1, self.reprst_D) # [B, reprst_H*reprst_W]
        return curr_states
    


class Predictor_2dCNN(nn.Module):
    def __init__(self, reprst_H, reprst_W):
        super().__init__()
        self.reprst_H = reprst_H
        self.reprst_W = reprst_W
        self.action_projector = nn.Sequential(
            nn.Linear(2, self.reprst_H * self.reprst_W),
            nn.ReLU()
        )
        # input: [B, 2, reprst_H, reprst_W]
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding=1), # ch1: prev_states, ch2: actions_proj
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.BatchNorm2d(64),
            # nn.Conv2d(in_channels=64, out_channels=2, kernel_size=3, padding=1)
        )
        # input: [B, 1, reprst_H, reprst_W]

    def forward(self, prev_states, actions):
        """
        Args:
            During training:
                prev_states: [B, reprst_H, reprst_W]
            During inference: will unroll the JEPA world model recurrently into the future, conditioned on "initial" observation and action sequence 
                prev_states: [B, reprst_H, reprst_W]
            actions: [B, 2]
        Output:
            curr_states: [B, reprst_H, reprst_W]
        """
        actions_proj = self.action_projector(actions).view(-1, self.reprst_H, self.reprst_W) # [B, reprst_H, reprst_W]
        input = torch.stack((prev_states, actions_proj), dim=1) # input: [B, 2, reprst_H, reprst_W]
        curr_states = self.cnn(input) # [B, 1, reprst_H, reprst_W]
        curr_states = curr_states.view(-1, self.reprst_H, self.reprst_W) # [B, reprst_H, reprst_W]
        return curr_states


class JEPAWorldModel(nn.Module):
    def __init__(self, encoder, encoder_target, predictor, device="cuda"):
        super().__init__()
        self.encoder = encoder # todo: same or not
        self.encoder_target = encoder_target # todo: same or not
        self.predictor = predictor
        # self.funct_distance = funct_distance
        self.device = device

    def forward(self, observs, actions):
        """
        Args:
            During training:
                observs: [B(batch size), T, Ch, H, W]
            During inference: will unroll the JEPA world model recurrently into the future, conditioned on "initial" observation and action sequence 
                observs: [B, 1, Ch, H, W]
            actions: [B, T-1, 2]
        Output:
            predictions: [B, T, D ("flattened" repr_dim)]
            targets: 
        """
        Bsize, T, _, _, _ = observs.shape
        pred_states = [] 
        target_states = []
        
        states_0 = self.encoder(observs[:, 0]) # states_0: [B, D], observs[:, 0]: [B, Ch, H, W]
        pred_states_1 = self.predictor(states_0, actions[:, 0]) # pred_states_1: [B, D]
        pred_states.append(pred_states_1) # [s1]
        target_states_1 = self.encoder_target(observs[:, 1]) # target_states_1: [B, D]
        target_states.append(target_states_1) # [s1']
        
        for t in range(1, T-1):
            pred_states_t = self.predictor(pred_states[t-1], actions[:, t])
            pred_states.append(pred_states_t) # [s1, s2]
            target_states_t = self.encoder_target(observs[:, t+1])
            target_states.append(target_states_t) # [s1', s2']

        return torch.stack(pred_states, dim=1), torch.stack(target_states, dim=1) # concatenate states of different timesteps => [B, T-1, D]


class Prober(torch.nn.Module):
    def __init__(
        self,
        embedding: int,
        arch: str,
        output_shape: List[int],
    ):
        super().__init__()
        self.output_dim = np.prod(output_shape)
        self.output_shape = output_shape
        self.arch = arch

        arch_list = list(map(int, arch.split("-"))) if arch != "" else []
        f = [embedding] + arch_list + [self.output_dim]
        layers = []
        for i in range(len(f) - 2):
            layers.append(torch.nn.Linear(f[i], f[i + 1]))
            layers.append(torch.nn.ReLU(True))
        layers.append(torch.nn.Linear(f[-2], f[-1]))
        self.prober = torch.nn.Sequential(*layers)

    def forward(self, e):
        output = self.prober(e)
        return output


class BarlowTwinsLoss(nn.Module):
    def __init__(self, lambda_=5e-3):
        """
        Barlow Twins Loss Module.

        Args:
            lambda_ (float): Scaling factor for the redundancy reduction term.
        """
        super(BarlowTwinsLoss, self).__init__()
        self.lambda_ = lambda_

    def forward(self, preds, targets):
        """
        Computes the Barlow Twins loss.

        Args:
            preds (torch.Tensor): Embeddings from the first view. Shape: (batch_size, T-1, embedding_dim).
            targets (torch.Tensor): Embeddings from the second view. Shape: (batch_size, T-1, embedding_dim).

        Returns:
            torch.tensor(np.mean(lt_loss))
        """
        batch_size, traj_length, embedding_dim = preds.shape
        total_loss = 0.0
        # lt_loss = []
        for t in range(traj_length):
            z1 = preds[:, t] # [batch_size, embedding_dim]
            z2 = preds[:, t]
        
            # Normalize embeddings
            z1 = F.normalize(z1, dim=1)
            z2 = F.normalize(z2, dim=1)
            
            # Cross-correlation matrix
            # batch_size = z1.size(0)
            c = (z1.T @ z2) / batch_size

            # Diagonal loss (invariance loss)
            identity_loss = torch.mean((torch.diag(c) - 1) ** 2)
    
            # Off-diagonal loss (redundancy reduction)
            off_diag = c - torch.eye(embedding_dim, device=c.device)
            off_diag_loss = torch.mean(off_diag ** 2)

            # Combined loss for this timestep
            timestep_loss = identity_loss + self.lambda_ * off_diag_loss
            total_loss += timestep_loss
    
            # # Identity matrix
            # on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()  # (Cii - 1)^2
            # off_diag = self.off_diagonal(c).pow_(2).sum()       # Cij^2 for i != j
    
            # # Total loss
            # loss = on_diag + self.lambda_ * off_diag
            # lt_loss.append(loss.item())
        return total_loss/traj_length


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

reprst_H=32
reprst_W=32
Enc = Encoder_ViT(reprst_H, reprst_W).to(device)
Enc_t = Encoder_ViT(reprst_H, reprst_W).to(device)
Pred = Predictor_1dCNN(reprst_H, reprst_W).to(device)
model = JEPAWorldModel(encoder=Enc, encoder_target=Enc_t, predictor=Pred).to(device)
# model.load_state_dict(torch.load("/content/best_model (1).pth", weights_only=True))

criterion = BarlowTwinsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=30, eta_min=1e-6)

dataset = create_wall_dataloader(data_path='./DL24FA/train', device=device, batch_size=64)

num_epochs = 10
min_loss = float('inf')
# step = 0
for epoch in tqdm(range(num_epochs), desc=f""):
    model.train()
    total_loss = 0
    for batch in tqdm(dataset, desc=""):
        observs = batch.states.to(device)
        actions = batch.actions.to(device)
        
        optimizer.zero_grad()
        
        pred_states, target_states = model(observs, actions)
        loss = criterion(pred_states, target_states) # mean of losses (across timesteps)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        # scheduler step

        # if step % 100 == 0:
        #     print(f"training loss {loss.item()}")

        # step += 1
    mean_loss = total_loss/len(dataset)
    print(f"Epoch: {epoch+1}, Training Loss: {mean_loss: .4f}")

    if mean_loss < min_loss:
        min_loss = mean_loss
        torch.save(model.state_dict(), "best_ViT_1dCNN.pth")
    
