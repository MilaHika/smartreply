from params import hp
import torch
from model import Net
from pytorch_pretrained_bert import BertTokenizer
from collections import OrderedDict
import pickle, re
import liwc
from collections import OrderedDict, Counter
from profanity_check import predict, predict_prob

import argparse
import json
from flask import Flask, request, jsonify

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
ckpt = 'model/smartreply.pt'

print("Wait... loading model")
model = Net(hp.n_classes)
model = model.to(device)
ckpt = torch.load(ckpt, map_location='cpu')

ckpt = OrderedDict([(k.replace("module.", ""), v) for k, v in ckpt.items()])
model.load_state_dict(ckpt)
print("Model loaded.")

print(f"Loading dictionaries ... from {hp.idx2phr}")
idx2phr = pickle.load(open(hp.idx2phr, 'rb'))

def prepare_inputs(context, tokenizer):
    tokens = tokenizer.tokenize(context)
    tokens = tokenizer.convert_tokens_to_ids(tokens)[-hp.max_span+2:]
    tokens = [101] + tokens + [102]
    tokens = torch.LongTensor(tokens)
    tokens = tokens.unsqueeze(0) 
    tokens = tokens.to("cpu")
    return tokens

def tokenize(text):
    for match in re.finditer(r'\w+', text, re.UNICODE):
        yield match.group(0)

parse, category_names = liwc.load_token_parser('data/LIWC2007_English100131.dic')

def polite_check(responses):
    polite_scores = {}
    for i, response in enumerate(responses):
        ht_tokens = tokenize(response)
        categories = Counter(category for token in ht_tokens for category in parse(token))
        cognitive = ['insight', 'discrep', 'tentat', ]
        cognitive_score = sum([categories[cog] for cog in cognitive])
        profanity_score = predict_prob([response])
        length_score = len(response.split())
        score = -0.6*profanity_score + 0.3*cognitive_score + 0.1*length_score
        polite_scores[response] = score[0]
    polite_scores = dict(sorted(polite_scores.items(), key=lambda item: item[1], reverse=True))
    return polite_scores

def suggest(context, tokenizer, model, idx2phr, ilist=False, polite=False, impolite=False, verbose=False):
    x = prepare_inputs(context, tokenizer)
    model.eval()
    with torch.no_grad():
        _, y_hat, y_hat_prob = model(x)
        y_hat = y_hat.to(device).numpy().flatten()  # (3)
        y_hat_prob = y_hat_prob.cpu().numpy().flatten()  # (3)
        y_hat_prob = [round(each, 2) for each in y_hat_prob]
        preds = [idx2phr.get(h, "None") for h in y_hat]
        probs = {}
        for i, pred in enumerate(preds):
            probs[pred] = y_hat_prob[i]
        polite_prebs = polite_check(preds)
        if polite:
            preds = list(polite_prebs.keys())[0:5]
        elif impolite:
            preds = list(polite_prebs.keys())[-5:]
        if ilist:
            preds = list(filter(lambda x: 'I' in x, preds))[0:5]
        if verbose:
            for pred in preds:
                if polite:
                    print(f'{pred} | probability: {probs[pred]} | politeness: {polite_prebs[pred]}')
                else:
                    print(f'{pred} | probability: {probs[pred]}')
        
        res = {}
        for pred in preds:
            if polite or impolite:
                res[pred] = (f'{probs[pred]:9f}', f'{polite_prebs[pred]:9f}')
            else:
                res[pred] = (f'{probs[pred]:9f}', f'-1')
        return res
        
# def suggest(context, tokenizer, model, idx2phr):
#     x = prepare_inputs(context, tokenizer)
#     model.eval()
#     with torch.no_grad():
#         _, y_hat, y_hat_prob = model(x)
#         y_hat = y_hat.to(device).numpy().flatten() 
#         y_hat_prob = y_hat_prob.cpu().numpy().flatten() 
#         y_hat_prob = [round(each, 2) for each in y_hat_prob]
#         preds = [idx2phr.get(h, "None") for h in y_hat]

#         res = {}
#         for i, pred in enumerate(preds):
#             res[pred] = str(y_hat_prob[i])
#         return res

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

@app.route('/predict', methods=['GET'])
def predict():
    context = request.args.get('context')
    ilist = str2bool(str(request.args.get('ilist')))
    polite = str2bool(str(request.args.get('polite')))
    impolite = str2bool(str(request.args.get('impolite')))
    res =  suggest(context, tokenizer, model, idx2phr, ilist=ilist, polite=polite, impolite=impolite, verbose=False)
    return jsonify(res)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=105)