class Params:
    n_classes = 100
    idx2phr = "data/idx2phr.pkl"
    max_span = 128 # maximum token length for context
    n_candidates = 20 # number of return predictions

hp = Params()