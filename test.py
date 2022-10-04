import requests

query = {
    'context': 'hello',
    'polite': True,
    'impolite': False,
    'ilist': False
}

def predict(query={'context': 'hello', 'polite': True, 'impolite': False}):
    r = requests.get('http://localhost:105/predict', params=query)
    # use v[0] to sort based on probability
    # use v[1] to sort based on politeness/impoliteness
    res = {k: float(v[1]) for k, v in r.json().items()}
    res = dict(sorted(res.items(), key=lambda item: item[1], reverse=True))
    print(res)
    return res

if __name__ == "__main__":
    predict()
