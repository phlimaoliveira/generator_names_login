from flask import Flask, jsonify

from eping_generator import EpingGenerator

app = Flask(__name__)

@app.route("/eping/generate/<nome_completo>", methods=["POST"])
def generateLogin(nome_completo):
    lista_logins = EpingGenerator.generateLogin(nome_completo)

    return jsonify(lista_logins)

@app.route("/eping/is_composite_name/<nome>", methods=["GET"])
def isCompositeName(nome):
    return EpingGenerator.isCompositeName(nome)

if __name__ == '__main__':
    app.run(debug=True)