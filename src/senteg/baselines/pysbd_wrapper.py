from pysbd import Segmenter
class PySBDSplitter:
    def __init__(self, lang="vi"):
        self.seg = Segmenter(language=lang, clean=False)
    def split(self, text: str):
        return self.seg.segment(text)
