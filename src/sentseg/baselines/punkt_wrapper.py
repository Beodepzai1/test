import nltk, io
nltk.download("punkt", quiet=True)

class PunktSplitter:
    """Huấn luyện nhanh Punkt trên chính đoạn văn (unsupervised)."""
    def __init__(self):
        pass
    def split(self, text: str):
        trainer = nltk.tokenize.punkt.PunktTrainer()
        trainer.train(text)
        tok = nltk.tokenize.punkt.PunktSentenceTokenizer(trainer.get_params())
        return tok.tokenize(text)
