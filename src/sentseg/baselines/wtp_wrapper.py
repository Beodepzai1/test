from wtpsplit import WtP


class WtPSplitter:
    def __init__(self, model: str = "wtp-bert-mini"):
        """Wrapper around :class:`wtpsplit.WtP` with a sane default model.

        The previously used ``wtp-xlmr-c3`` model has been removed from
        HuggingFace, causing downloads to fail.  ``wtp-bert-mini`` is still
        available and lightweight enough for most use cases, so we adopt it as
        the new default.
        """
        self.wtp = WtP(model)

    def split(self, text: str, lang: str = "vi"):
        return self.wtp.split(text, lang_code=lang)
