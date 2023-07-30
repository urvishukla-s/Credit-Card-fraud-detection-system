"""
Main App.py where Business Logic Goes
It Consist of ML Model integration with Blockchain Components
"""
import ast
import datetime
import hashlib
import json
import pickle
import werkzeug
import numpy as np
from flask import Flask, abort, render_template, request#, jsonify

FILENAME = "lr_model.pickle"
with open(FILENAME, "rb") as file_handle:
    clf=pickle.load(file_handle)
app = Flask(__name__)


class Blockchain:
    """ Class to Implement Blockchain """
    def __init__(self):
        """ initialization """
        self.b_chain = []
        with open("static\ledger.txt", encoding="utf-8") as file_in:
            for line in file_in:
                self.b_chain.append(ast.literal_eval(line))

    def get_index(self):
        """ Get the Ledger Index """
        idx = self.b_chain[-1]['index']
        return idx

    def create_block(self,idx, proof, previous_hash, transaction):
        """ create a block """
        block = {'index': idx,
                'transaction': transaction,
                'timestamp': str(datetime.datetime.now()),
                'proof': proof,
                'previous_hash': previous_hash}
        self.b_chain.append(block)
        with open("static\ledger.txt",'a+',encoding="utf-8") as fileptr:
            fileptr.write(str(block)+'\n')
        return block

    def print_previous_block(self):
        """ Return last block """
        return self.b_chain[-1]

    def proof_of_work(self, previous_proof):
        """ Calculate PoW """
        new_proof = 1
        check_proof = False

        while check_proof is False:
            hash_operation = hashlib.sha256(
                str(new_proof**2 - previous_proof**2).encode()).hexdigest()
            if hash_operation[:5] == '00000':
                check_proof = True
            else:
                new_proof += 1

        return new_proof

    def hash(self, block):
        """ Return Hash """
        encoded_block = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(encoded_block).hexdigest()

    def chain_valid(self, b_chain):
        """ Check Validity of Block """
        previous_block = b_chain[0]
        block_index = 1

        while block_index < len(b_chain):
            block = b_chain[block_index]
            if block['previous_hash'] != self.hash(previous_block):
                return False

            previous_proof = previous_block['proof']
            proof = block['proof']
            hash_operation = hashlib.sha256(
                str(proof**2 - previous_proof**2).encode()).hexdigest()

            if hash_operation[:5] != '00000':
                return False
            previous_block = block
            block_index += 1

        return True

blockchain = Blockchain()

@app.route('/')
def home():
    """ render home page """
    return render_template('cc.html')

@app.route('/predict', methods=['POST'])
def predict():
    """ prediction module """
    if request.method == 'POST':
        m_e = request.form['message']
        message = [float(x) for x in m_e.split()]
        vect = np.array(message).reshape(1, -1)
        my_prediction = clf.predict(vect)

        if ((my_prediction == 0) and valid_block()):
            transaction = m_e
            previous_block = blockchain.print_previous_block()
            previous_proof = previous_block['proof']
            proof = blockchain.proof_of_work(previous_proof)
            previous_hash = blockchain.hash(previous_block)
            idx = blockchain.get_index() + 1
            blockchain.create_block(idx, proof, previous_hash, transaction)
    return render_template('result.html', prediction=my_prediction)

@app.route("/ledger")
def ledger():
    """  Display Blockchain """
    return render_template('ledger.html',data = blockchain.b_chain)

def valid_block():
    """ Check Validity of Existing Chain """
    return blockchain.chain_valid(blockchain.b_chain)
    #if valid:
        #response = {'message': 'The Blockchain is valid.'}
    #else:
        #response = {'message': 'The Blockchain is not valid.'}
    #return jsonify(response), 200

@app.route("/simulate404")
def simulate404():
    """ Custom Error Handling """
    abort(404)
    return render_template("html.html")

@app.route("/simulate500")
def simulate500():
    """ Simulate error 500 """
    abort(500)
    return render_template("html.html")


@app.route("/about")
def about():
    """ Redender About us """
    return render_template("about.html")

@app.route("/cc")
def c_c():
    """ CC rendering """
    return render_template("home.html")

@app.route("/datasets")
def datasets():
    """ display sample Dataset """
    return render_template("csv.html")

@app.errorhandler(404)
def not_found_error(error):
    """ 404 """
    return render_template('html.html'), 404

@app.errorhandler(werkzeug.exceptions.HTTPException)
def internal_error(error):
    """ Internal Error """
    return render_template('html.html'), 500

if __name__ == '__main__':
    app.run(debug=False)
