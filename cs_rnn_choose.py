import joblib


def compile_feats(logprob, entropy, confidence, cs_logprob, predicted):
    retval = {}
    retval['logprob'] = logprob
    retval['entropy'] = entropy
    retval['confidence'] = confidence
    retval['predicted'] = predicted
    if cs_logprob is not None:
        retval['cs_logprob'] = cs_logprob
    else:
        retval['no_cs_prob'] = True

    return retval


class CS_RNN_chooser:
    def __init__(self, logistic_fn, vectorizer_fn):
        self.cs_rnn_mdl = joblib.load(logistic_fn)
        self.vectorizer = joblib.load(vectorizer_fn)

    def switch_to_RNN(self, feature_dict):
        feats = self.vectorizer.transform([feature_dict])
        result = self.cs_rnn_mdl.predict(feats)[0]
        return result
