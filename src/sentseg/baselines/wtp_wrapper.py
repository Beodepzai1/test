from wtpsplit import WtP


class WtPSplitter:
    def __init__(self, model: str = "wtp-xlmr-c3"):
        self.wtp = WtP(model)

    def split(self, text: str, lang: str = "vi"):
        return self.wtp.split(text, lang_code=lang)
