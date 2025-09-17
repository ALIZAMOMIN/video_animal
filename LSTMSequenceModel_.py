#lstm sequence model
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMSequenceModel(nn.Module):
    def __init__(self, num_features, max_seq_length, num_classes):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=num_features, hidden_size=16, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=16, hidden_size=8, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, num_classes)
    def forward(self, x, mask=None):
        # x: (batch_size, seq_len, num_features)
        lengths = mask.sum(dim=1) if mask is not None else torch.full((x.size(0),), x.size(1), dtype=torch.long).to(x.device)
        lengths = torch.clamp(lengths, min=1)  # Prevent zero lengths
        # Pack padded sequence for the LSTM
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (h1, _) = self.lstm1(packed)
        packed_output, (h2, _) = self.lstm2(packed_output)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, total_length=x.size(1))

        # Get the last valid output in each sequence according to actual length
        idx = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, output.shape[2])
        last_outputs = output.gather(1, idx).squeeze(1)
        x = self.dropout(last_outputs)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        #print(x.shape) 16,2
        return x  # logits

