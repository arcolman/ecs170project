"""Word-level recurrent language model for joke generation.

cell='rnn' | 'lstm' | 'gru' swaps only the recurrent layer (nn.RNN, nn.LSTM, nn.GRU); embedding, output layer,
optimizer and every hyperparameter stay the same.
"""
import time
import torch
import torch.nn as nn

CELLS = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}


def pick_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class Method_RNN_Generation(nn.Module):
    def __init__(self, vocab_size, cell='rnn', emb_dim=64, hidden_dim=128, num_layers=1, dropout=0.0):
        super().__init__()
        self.cell = cell
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = CELLS[cell](emb_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                               dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, h=None):
        out, h = self.rnn(self.drop(self.embedding(x)), h)
        return self.fc(self.drop(out)), h

    def fit(self, jokes, stoi, device, epochs=30, batch_size=32, lr=3e-3, seed=1, log=print):
        """Teacher-forced next-word prediction on each joke (padded batches, pad positions ignored)."""
        seqs = [torch.tensor([stoi[t] for t in j]) for j in jokes]
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        g = torch.Generator().manual_seed(seed)
        history, start = [], time.time()
        self.to(device)
        for ep in range(epochs):
            self.train()
            total, count = 0.0, 0
            for idx in torch.randperm(len(seqs), generator=g).split(batch_size):
                batch = nn.utils.rnn.pad_sequence([seqs[i] for i in idx], batch_first=True).to(device)
                x, y = batch[:, :-1], batch[:, 1:]
                logits, _ = self(x)
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 5.0)
                opt.step()
                n = (y != 0).sum().item()
                total += loss.item() * n
                count += n
            history.append(total / count)
            log(f'  [{self.cell}] epoch {ep + 1:2d}/{epochs}  train loss {history[-1]:.4f}')
        return history, time.time() - start

    @torch.no_grad()
    def generate(self, prompt_ids, n_tokens, device, temperature=0.8, seed=1):
        """Feeds the prompt, then samples n_tokens words. <eos> is kept as a joke separator and generation continues."""
        self.eval()
        g = torch.Generator(device='cpu').manual_seed(seed)
        logits, h = self(torch.tensor([prompt_ids], device=device))
        out = []
        for _ in range(n_tokens):
            probs = torch.softmax(logits[0, -1].float().cpu() / temperature, dim=-1)
            probs[0] = 0.0  # never emit <pad>
            nxt = torch.multinomial(probs, 1, generator=g).item()
            out.append(nxt)
            logits, h = self(torch.tensor([[nxt]], device=device), h)
        return out
