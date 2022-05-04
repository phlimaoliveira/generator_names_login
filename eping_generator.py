import random, json
from flask import jsonify

import torch

from machine_learning.model import NeuralNet
from machine_learning.nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('machine_learning/names.json', 'r', encoding="utf-8") as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

class EpingGenerator:
    def generateLogin(nome_completo):
        return str('login.unico')

    def gender_features(word):
        return {'last_letter': word[-1]}

    def isCompositeName(name):
        sentence = tokenize(name)
        if len(sentence) > 1:
            X = bag_of_words(sentence, all_words)
            X = X.reshape(1, X.shape[0])
            X = torch.from_numpy(X).to(device)

            output = model(X)
            _, predicted = torch.max(output, dim=1)

            tag = tags[predicted.item()]

            probs = torch.softmax(output, dim=1)
            prob = probs[0][predicted.item()]
            if prob.item() > 0.99:
                for intent in intents['intents']:
                    if tag == intent["tag"]:
                        return jsonify(f"{random.choice(intent['responses'])}")
            return jsonify(f"NÃO É UM NOME COMPOSTO")
        else:
            return jsonify(f"NÃO É UM NOME COMPOSTO")