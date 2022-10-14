from flask import Flask, jsonify

from generator_names import NamesGenerator

app = Flask(__name__)

@app.route("/generate/<nome_completo>", methods=["POST"])
def generateLogin(nome_completo):
    lista_logins = NamesGenerator.generate_login(nome_completo)

    return jsonify(lista_logins)

@app.route("/is_composite_name/<nome>", methods=["GET"])
def isCompositeName(nome):
    return NamesGenerator.is_composite_name(nome)

if __name__ == '__main__':
    app.run(debug=True)