import sklearn_crfsuite
from sentseg.features import sent2features, sent2labels

def train(train_sents, c1=0.1, c2=0.1, max_iter=200):
    X = [sent2features(s) for s in train_sents]
    y = [sent2labels(s) for s in train_sents]
    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs", c1=c1, c2=c2,
        max_iterations=max_iter, all_possible_transitions=True
    )
    crf.fit(X, y)
    return crf
