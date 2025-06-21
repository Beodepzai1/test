from __future__ import annotations
import importlib
from typing import Iterable, Tuple


def _load_torch():
    try:
        torch = importlib.import_module("torch")
        nn = importlib.import_module("torch.nn")
        F = importlib.import_module("torch.nn.functional")
        return torch, nn, F
    except Exception as e:
        raise ImportError("PyTorch is required for this module") from e


def build_textcnn(vocab_size: int, num_classes: int, embed_dim: int = 128,
                  kernel_sizes: Iterable[int] = (3, 4, 5), num_filters: int = 100,
                  dropout: float = 0.5):
    torch, nn, F = _load_torch()

    class TextCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.convs = nn.ModuleList([
                nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes
            ])
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

        def forward(self, x):
            x = self.embedding(x).permute(0, 2, 1)
            conved = [F.relu(c(x)).max(dim=2)[0] for c in self.convs]
            out = torch.cat(conved, dim=1)
            out = self.dropout(out)
            return self.fc(out)

    return TextCNN()


def build_gru(vocab_size: int, num_classes: int, embed_dim: int = 128,
              hidden_dim: int = 128, num_layers: int = 1,
              bidirectional: bool = True, dropout: float = 0.5):
    torch, nn, F = _load_torch()

    class GRUClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers,
                              batch_first=True, bidirectional=bidirectional,
                              dropout=dropout if num_layers > 1 else 0)
            direc = 2 if bidirectional else 1
            self.fc = nn.Linear(hidden_dim * direc, num_classes)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            x = self.dropout(self.embedding(x))
            _, h = self.rnn(x)
            if bidirectional:
                h = torch.cat([h[-2], h[-1]], dim=1)
            else:
                h = h[-1]
            return self.fc(h)

    return GRUClassifier()


def build_bert(model_name: str = "bert-base-uncased", num_classes: int = 3):
    torch, nn, _ = _load_torch()
    transformers = importlib.import_module("transformers")
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_classes
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer
