try:
    from wtpsplit import WtP
except Exception:  # pragma: no cover - optional dependency
    WtP = None

class WtPSplitter:
    def __init__(self, model="wtp-xlmr-c3", device="cpu"):
        if WtP is None:
            raise ImportError("wtpsplit not installed")
        self.wtp = WtP(model, device=device)
    def split(self, text: str, lang="vi"):
        return self.wtp.split(text, lang_code=lang)
