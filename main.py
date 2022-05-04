from flask import Flask, jsonify

import random
import json

import torch

from machine_learning.model import NeuralNet
from machine_learning.nltk_utils import bag_of_words, tokenize

from eping_generator import EpingGenerator

app = Flask(__name__)
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

@app.route("/eping/generate/<nome_completo>", methods=["POST"])
def generateLogin(nome_completo):
    login_unico = EpingGenerator.generateLogin(nome_completo)

    return jsonify({"login": login_unico})

@app.route("/eping/is_composite_name/<nome>", methods=["GET"])
def isCompositeName(nome):
    return EpingGenerator.isCompositeName(nome)

if __name__ == '__main__':
    app.run(debug=True)