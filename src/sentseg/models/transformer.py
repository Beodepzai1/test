from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

class PhoBERTSegmenter:
    """Fine‑tune / infer sentence boundaries B=1, I=0"""
    def __init__(self, model_name="vinai/phobert-base", num_labels=2):
        self.tk = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict(self, text: str):
        self.model.eval()
        ids = self.tk(text, return_tensors="pt", truncation=True)
        input_ids = ids["input_ids"]
        ids = {k: v.to(self.device) for k, v in ids.items()}
        with torch.no_grad():
            out = self.model(**ids).logits.argmax(-1).squeeze().cpu()
        toks = self.tk.convert_ids_to_tokens(input_ids.squeeze())
        sent, sents = [], []
        for tok, lab in zip(toks, out.tolist()):
            if tok in ["<s>", "</s>"]: continue
            sent.append(tok)
            if lab == 1:
                sents.append(" ".join(sent)); sent = []
        if sent: sents.append(" ".join(sent))
        return sents
