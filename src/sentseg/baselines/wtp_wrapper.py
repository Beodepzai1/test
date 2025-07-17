from wtpsplit import WtP


class WtPSplitter:
    def __init__(self, model: str = "wtp-bert-mini"):
        """Wrapper around :class:`wtpsplit.WtP`."""
        self.wtp = WtP(model)

    def split(self, text: str, lang: str = "vi"):
        return self.wtp.split(text, lang_code=lang)
