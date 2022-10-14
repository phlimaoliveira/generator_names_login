import random, json
from flask import jsonify

import torch
import nltk
from string import punctuation
import unidecode
from itertools import permutations
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

class NamesGenerator:

    def tratar(nome_completo):
        nome_lower_case = str(nome_completo).lower()
        remocao_pontuacao = ''.join([letra for letra in nome_lower_case if letra not in punctuation])
        nome_sem_acentos = unidecode.unidecode(remocao_pontuacao)
        nome_tokenizado = nltk.word_tokenize(nome_sem_acentos)

        stopwords = nltk.corpus.stopwords.words('portuguese')

        nome_tratado = [p for p in nome_tokenizado if p not in stopwords]
        return nome_tratado

    def generate_login(nome_completo):
        separator = "."
        nome_tratado = NamesGenerator.tratar(nome_completo)
        temp = permutations(nome_tratado, 2)
        lista = []
        for i in list(temp):
            loginComPonto = separator.join(i)
            lista.append(loginComPonto)

        for i in range(1, 50):
            loginComNumero = lista[0] + "." + str(i)
            lista.append(loginComNumero)

        return lista

    def is_composite_name(name):
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
                        return jsonify(f"{random.choice(intent['responses'])}") # True
            return jsonify(f"NÃO É UM NOME COMPOSTO") # False
        else:
            return jsonify(f"NÃO É UM NOME COMPOSTO") # False