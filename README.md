Install the necessary packages.
```
pip install requirements.txt
```
The smartreply model can be accessed [here](https://drive.google.com/drive/folders/1gWftiE0llzhiQ5Og8hyh-_XTbFKSEr7s?usp=sharing). Please create a directory titled `model` and add the `smartreply.pt` file to it.

To run the API, start the python script for `api.py`.
```
python api.py
```

The current api is run on localhost port **105**. Feel free to change this.


## **Predict** | API Documentation
The api has a single command **predict** which returns 5 of the predicted replies from the trained smart reply model in reponse to the given text input. 

```http
GET /predict
```

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `context` | `string` | **Required**. Text input which you want a smart reply to. |
| `ilist` | `boolean` | **Required**. Boolean input to indicate if the `I` based rule should be turned on.
| `polite` | `boolean` | **Required**. Boolean input to indicate if the politeness based rule should be turned on.
| `impolite` | `boolean` | **Required**. Boolean input to indicate if the impoliteness based rule should be turned on.

> Note that the API end point can also allow for verbose results and print the prediction probability and politeness score of responses. To allow for this, simply navigate to the file `api.py` and set the param `verbose=True`.

```
suggest(context, tokenizer, model, idx2phr, ilist=False, polite=False, impolite=False, verbose=False)
```

## **Predict** | API Reponse
The API end point returns in json format where the keys are defined to be the string replies predicted by the smart reply model. The first value is specified to be the prediction probabilities and the second value is specified to be the politness or 
impolitness scored which is calculated based on use of mediating words, profanities, 
and length/curtness of response. If response does not use the rule-based approach 
of politeness or impoliteness, the second value defaults to -1.

```json
{
    'response1': ('0.25', '0.64'),
    'response2': ('0.15', '0.25'),
    'response3': ('0.03', '0.25'),
    'response4': ('0.01', '0.12'),
    'response5': ('0.01', '0.05'),
}
```

> Note that since the API end point respsonse returns in json format, the results 
are and unordered dictionary and needs to be sorted.

## Example
A working example is provided in the `test.py` file to showcase how to call the **predict** api command. An example query is listed below.

```python
query = {
    'context': 'hello', 
    'polite': True, 
    'impolite': False,
    'ilist': False
}
```
