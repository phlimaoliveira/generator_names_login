from flask import Flask, request, jsonify
from eping_generator import EpingGenerator

app = Flask(__name__)

@app.route("/eping/generate/<nome_completo>", methods=["POST"])
def generateLogin(nome_completo):
    login_unico = EpingGenerator.generateLogin(nome_completo)

    return jsonify({"login": login_unico})

if __name__ == '__main__':
    app.run(debug=True)