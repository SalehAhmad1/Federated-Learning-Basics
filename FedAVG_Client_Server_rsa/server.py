from model import *
from model_utils import *

import pickle
import base64
from cryptography.hazmat.primitives.asymmetric import rsa

import warnings
warnings.filterwarnings('ignore')

from flask import Flask, request, jsonify, Response, send_file, make_response

N = 2 #Number fo clients

app = Flask(__name__)
server_model = ServerModel(num_features=24, num_labels=5)

'''Generating Keys for RSA for the server'''
private_key_rsa = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048,
    backend=default_backend()
)
public_key_rsa = private_key_rsa.public_key()

'''Serializing Keys for RSA for the server'''
public_key_bytes_rsa = public_key_rsa.public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo
)

# A list to store the weights from each client
weights_list = {}

@app.route('/key', methods=['GET'])
def get_key():
    '''Get server's public key for RSA'''
    return jsonify({'key': public_key_bytes_rsa.decode('utf-8')})

@app.route('/FED', methods=['POST'])
def receive_weights():
    '''Perform aggregation and send the aggregated weights to the client'''
    global weights_list
    global private_key_rsa

    # Assuming the weights are sent as a JSON object
    base64_weights = request.json['encrypted_base64_weights']
    client_public_key_bytes_rsa = request.json['key'].encode('utf-8')

    # Decrypt the weights
    decrypted_weights = decrypt(base64_weights, private_key_rsa)

    # Convert the weights from a dictionary to a PyTorch state dict
    weights_dict = {k: torch.tensor(v) for k, v in decrypted_weights.items()}
    weights_list[client_public_key_bytes_rsa] = weights_dict

    return jsonify({'message': 'Weights received successfully'}), 200

@app.route('/check_for_wait', methods=['GET'])
def check_weights():
    global weights_list
    global N
    if len(weights_list) == N:
        return jsonify({'message': 'All weights received'}), 200
    else:
        return jsonify({'message': 'Not all weights received'}), 400
    
@app.route('/aggregate', methods=['POST'])
def aggregate_weights():
    global weights_list
    '''Check if all weights have been received'''
    if len(weights_list) == 0:
        return_message = jsonify({'message': 'No weights to aggregate'})
        return return_message, 400
    
    client_public_key_bytes_rsa = request.json['key'].encode('utf-8')
    client_public_key_rsa = serialization.load_pem_public_key(client_public_key_bytes_rsa, backend=default_backend())

    # Aggregating weights here, by averaging them
    aggregated_weights = {}
    for weights in weights_list.values():
        for k, v in weights.items():
            if k in aggregated_weights:
                aggregated_weights[k] += v
            else:
                aggregated_weights[k] = v
    for k, v in aggregated_weights.items():
        aggregated_weights[k] = v / N

    # Encrypt the model weights
    base64_weights = encrypt(aggregated_weights, client_public_key_rsa)

    # Convert the aggregated weights back to a dictionary
    return jsonify({'base64_weights': base64_weights}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)