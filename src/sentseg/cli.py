import argparse, yaml, csv, random
from pathlib import Path
from sentseg import dataset as ds, trainer, evaluator
from sentseg.features import sent2features, sent2labels
from sentseg.baseline import split as regex_split
from sentseg.baselines import (
    pysbd_wrapper, punkt_wrapper, wtp_wrapper
)

def run_baseline(baseline, cfg):
    txt = Path(cfg["data"]["test_path"]).read_text(encoding="utf-8")[:1000]
    if baseline == "regex":
        print(regex_split(txt)); return
    if baseline == "pysbd":
        print(pysbd_wrapper.PySBDSplitter().split(txt)); return
    if baseline == "punkt":
        print(punkt_wrapper.PunktSplitter().split(txt)); return
    if baseline == "wtp":
        print(wtp_wrapper.WtPSplitter().split(txt)); return
    if baseline == "wtp_finetune":
        # few‑shot on dev (n câu)
        from wtpsplit import WtP
        n = cfg["wtp"]["finetune_sentences"]
        dev_df = ds.load(cfg)[1].head(n)
        wtp = WtP(cfg["wtp"]["model_name"])
        sents = dev_df["free_text"].tolist()
        labels = ["\n".join(regex_split(t)) for t in sents]
        wtp.finetune(list(zip(sents, labels)), lang_code="vi")
        print(wtp.split(txt, lang_code="vi")); return
    raise ValueError("unknown baseline")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--baseline",
                    choices=["regex","crf","phobert",
                             "pysbd","punkt","wtp","wtp_finetune"],
                    default="regex")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    if args.baseline in ["crf", "phobert"]:
        train_df, dev_df, test_df = ds.prepare(cfg)
        if args.baseline == "crf":
            model = trainer.train_crf(cfg)
            # evaluate on dev and test
            def _df2sents(df):
                sents = []
                for txt in df["free_text"].astype(str):
                    for sent in ds._split_by_punc(txt):
                        toks = ds._sent2tokens(sent)
                        sents.append([(tok, "B" if i == len(toks)-1 else "I")
                                      for i, tok in enumerate(toks)])
                return sents
            dev_s = _df2sents(dev_df)
            test_s = _df2sents(test_df)
            X_dev = [sent2features(s) for s in dev_s]
            y_dev = [sent2labels(s) for s in dev_s]
            X_test = [sent2features(s) for s in test_s]
            y_test = [sent2labels(s) for s in test_s]
            print("Dev:", evaluator.evaluate_crf(model, X_dev, y_dev))
            print("Test:", evaluator.evaluate_crf(model, X_test, y_test))
        else:
            trainer.train_transformer(cfg)
    else:
        run_baseline(args.baseline, cfg)

if __name__ == "__main__":
    main()
