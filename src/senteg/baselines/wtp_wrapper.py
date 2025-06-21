from wtpsplit import WtP

class WtPSplitter:
    def __init__(self, model="wtp-xlmr-c3", device="cpu"):
        self.wtp = WtP(model, device=device)
    def split(self, text: str, lang="vi"):
        return self.wtp.split(text, lang_code=lang)
