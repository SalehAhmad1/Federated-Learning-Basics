from model import *
from model_utils import *

import pickle
import base64

import warnings
warnings.filterwarnings('ignore')

from flask import Flask, request, jsonify, Response, send_file, make_response

app = Flask(__name__)
server_model = ServerModel(num_features=24, num_labels=5)

@app.route('/update_server', methods=['POST','GET'])
def receive_client_model():
    '''
    This endpoint will receive:
    - Model Weights in bytes
    The endpoint will update it's weights by averaging and then return
    '''
    encrypted_weights = request.json['encrypted_weights']
    key = request.json['key']

    decrypted_weights = decrypt_weights_AES(encrypted_weights, key)

    client_model_temp = ClientModel(num_features=24, num_labels=5)
    client_model_temp.load_state_dict(decrypted_weights)

    All_Models = [server_model, client_model_temp]
    server_model.load_state_dict(average_weights(All_Models))

    encrypted_weights, key = encrypt_weights_AES(server_model)
    concat_bytes = b'--'.join([val for key,val in encrypted_weights.items()])
    concat_bytes_with_key = concat_bytes + key
    # serialized_weights = pickle.dumps(concat_bytes_with_key)
    base64_weights = base64.b64encode(concat_bytes_with_key).decode('utf-8')
    return base64_weights

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
