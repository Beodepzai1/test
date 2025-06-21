def token2features(sent, i):
    word = sent[i][0]
    feats = {
        "bias": 1.0,
        "word.lower": word.lower(),
        "word.isupper": word.isupper(),
        "word.istitle": word.istitle(),
        "word.isdigit": word.isdigit(),
    }
    if i > 0:
        prev = sent[i-1][0]
        feats.update({
            "-1:word.lower": prev.lower(),
            "-1:isupper": prev.isupper(),
            "-1:istitle": prev.istitle(),
        })
    else:
        feats["BOS"] = True
    if i < len(sent)-1:
        nxt = sent[i+1][0]
        feats.update({
            "+1:word.lower": nxt.lower(),
            "+1:isupper": nxt.isupper(),
            "+1:istitle": nxt.istitle(),
        })
    else:
        feats["EOS"] = True
    return feats

def sent2features(sent): return [token2features(sent, i) for i in range(len(sent))]
def sent2labels(sent):   return [lab for _, lab in sent]
