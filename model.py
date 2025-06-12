# Define the model
import torch
import torch.nn as nn


class LSTM(nn.Module):
    """
    This LSTM predicts only one step ahead. It is not usable for sequence-to-sequence.
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Fully connected layer
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # out stands for hidden state h in every time step
        # _ stands for cell state c in every time step
        out, _ = self.lstm(x)

        # out[:, -1, :] selects the hidden state of the last time step for each sequence in the batch.
        # self.linear maps the hidden state to the output dimesnion feature
        out = self.linear(out[:, -1, :])
        return out


class LongLSTM(nn.Module):
    """
    This LSTM outputs latent vector after encoding the whole sequence
    and decodes the original sequence from the latent vector.
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size, pred_steps=1, latent_size=8):
        super(LongLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.pred_steps = pred_steps

        # Fully connected layer
        self.neck_enc = nn.Linear(hidden_size * 2, latent_size)
        self.neck_dec = nn.Linear(latent_size, hidden_size * 2)

        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Arguments:
            x: torch.tensor (b, n_steps, feat)

        Returns:
            out torch.tensor (B, T, output_size), latent torch.tensor (B, latent_size)
        """

        # hidden state is the last hidden state, shape (1, B, hidden_size)
        # out stands for hidden state h in every time step, shape (B, T, hidden_size)

        # encode
        out, (hidden, cell) = self.lstm(x)

        # linear dimension reduction
        state = torch.cat((hidden, cell), dim=2)  # (1, B, hidden_size*2)
        bottleneck = self.neck_enc(state)  # (1, B, latent_size)
        state = self.neck_dec(bottleneck)  # (1, B, hidden_size*2)

        # === decode

        # input
        B = x.shape[0]
        fake_input = torch.zeros((B, self.pred_steps, self.hidden_size)).to(x.device)

        # hidden state and cell state
        hidden = state[:, :, :self.hidden_size]
        cell = state[:, :, self.hidden_size:]

        # decode
        out, _ = self.decoder(fake_input, (hidden, cell))  # hidden has shape (B, pred_steps, hidden_size)

        # self.linear maps the hidden state to the output dimesnion feature
        out = self.linear(out)
        return (out, bottleneck[0])
