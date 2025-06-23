import pickle
from pathlib import Path
from sentseg.dataset import _sent2tokens
from sentseg.features import sent2features

class CRFSplitter:
    """Sentence splitter using a pre‑trained CRF model."""
    def __init__(self, model_path: str | Path):
        path = Path(model_path)
        self.model = pickle.loads(path.read_bytes())

    def split(self, text: str):
        tokens = _sent2tokens(text)
        if not tokens:
            return []
        feats = sent2features([(tok, "I") for tok in tokens])
        labels = self.model.predict([feats])[0]
        sents, buf = [], []
        for tok, lab in zip(tokens, labels):
            buf.append(tok)
            if lab == "B":
                sents.append(" ".join(buf))
                buf = []
        if buf:
            sents.append(" ".join(buf))
        return sents
