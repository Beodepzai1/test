import argparse, yaml, csv, random
from pathlib import Path
from sentseg import dataset as ds, trainer, evaluator
from sentseg.features import sent2features, sent2labels
from sentseg.baseline import split as regex_split
from sentseg.baselines import (
    pysbd_wrapper,
    punkt_wrapper,
    wtp_wrapper,
)

def run_baseline(baseline, cfg):
    _, dev_df, test_df = ds.load(cfg)

    if baseline == "regex":
        splitter = regex_split
    elif baseline == "pysbd":
        splitter = pysbd_wrapper.PySBDSplitter().split
    elif baseline == "punkt":
        splitter = punkt_wrapper.PunktSplitter().split
    elif baseline == "wtp":
        splitter = wtp_wrapper.WtPSplitter().split
    elif baseline == "wtp_finetune":
        from wtpsplit import WtP
        n = cfg["wtp"]["finetune_sentences"]
        dev_head = dev_df.head(n)
        wtp = WtP(cfg["wtp"]["model_name"])
        sents = dev_head["free_text"].tolist()
        labels = ["\n".join(regex_split(t)) for t in sents]
        wtp.finetune(list(zip(sents, labels)), lang_code="vi")
        splitter = lambda txt: wtp.split(txt, lang_code="vi")
    else:
        raise ValueError("unknown baseline")

    dev_res = evaluator.evaluate_split(splitter, dev_df)
    test_res = evaluator.evaluate_split(splitter, test_df)
    print(f"Dev F1={dev_res['f1']:.4f} Acc={dev_res['accuracy']:.4f}")
    print(f"Test F1={test_res['f1']:.4f} Acc={test_res['accuracy']:.4f}")

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
            dev_s = ds.df2sents(dev_df)
            test_s = ds.df2sents(test_df)
            X_dev = [sent2features(s) for s in dev_s]
            y_dev = [sent2labels(s) for s in dev_s]
            X_test = [sent2features(s) for s in test_s]
            y_test = [sent2labels(s) for s in test_s]
            dev_res = evaluator.evaluate_crf(model, X_dev, y_dev)
            test_res = evaluator.evaluate_crf(model, X_test, y_test)
            print(f"Dev F1={dev_res['f1']:.4f} Acc={dev_res['accuracy']:.4f}")
            print(f"Test F1={test_res['f1']:.4f} Acc={test_res['accuracy']:.4f}")
        else:
            trainer.train_transformer(cfg)
    else:
        run_baseline(args.baseline, cfg)

if __name__ == "__main__":
    main()
