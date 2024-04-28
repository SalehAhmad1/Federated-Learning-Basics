import torch
import torch.nn as nn
import torch.optim as optim

import pickle
import os
import base64
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives import padding as sym_padding

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def average_weights(models):        
    avg_state_dict = {}
    for key in models[0].state_dict().keys():
        avg_state_dict[key] = torch.stack([model.state_dict()[key] for model in models], 0).mean(0)
    return avg_state_dict

def encrypt(data, public_key):
    # Encrypt the model weights
    encrypted_weights = {}
    for k, v in data.items():
        v_bytes = pickle.dumps(v)
        # Generate a random AES key
        aes_key = os.urandom(32)

        # Encrypt the message using AES
        cipher = Cipher(algorithms.AES(aes_key), modes.ECB(), backend=default_backend())
        encryptor = cipher.encryptor()
        padder = sym_padding.PKCS7(algorithms.AES.block_size).padder()
        padded_message = padder.update(v_bytes) + padder.finalize()
        ciphertext_aes = encryptor.update(padded_message) + encryptor.finalize()

        # Encrypt the AES key using RSA
        ciphertext_rsa = public_key.encrypt(
            aes_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

        # Store encrypted weights
        encrypted_weights[k] = {
            "ciphertext_aes": ciphertext_aes,
            "ciphertext_rsa": ciphertext_rsa,
        }

    serialized_weights = pickle.dumps(encrypted_weights)
   # Convert bytes to Base64 string
    return base64.b64encode(serialized_weights).decode('utf-8')

def decrypt(base64_weights, private_key_rsa):
    # Convert Base64 string to bytes and deserialize
    serialized_weights = base64.b64decode(base64_weights)
    encrypted_weights = pickle.loads(serialized_weights)

    # Decrypt the AES key using RSA
    decrypted_weights = {}
    for k, v in encrypted_weights.items():
        ciphertext_rsa = v['ciphertext_rsa']
        aes_key = private_key_rsa.decrypt(
            ciphertext_rsa,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

        # Decrypt the message using AES
        ciphertext_aes = v['ciphertext_aes']
        cipher = Cipher(algorithms.AES(aes_key), modes.ECB(), backend=default_backend())
        decryptor = cipher.decryptor()
        unpadder = sym_padding.PKCS7(algorithms.AES.block_size).unpadder()
        decrypted_message_padded = decryptor.update(ciphertext_aes) + decryptor.finalize()
        decrypted_message = unpadder.update(decrypted_message_padded) + unpadder.finalize()

        # Deserialize the decrypted message
        original_message = pickle.loads(decrypted_message)
        decrypted_weights[k] = original_message

    return decrypted_weights

def save_model(model, filepath='./ServerModel.pth'):
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to '{filepath}'")

def load_model(model_class, args, filepath='./ServerModel.pth'):
    model = model_class(*args)
    model.load_state_dict(torch.load(filepath))
    model.eval()
    print(f"Model loaded from '{filepath}'")
    return model

def infer(model, inputs, idx2str):
    # Convert inputs to torch.Tensor if not already
    if not isinstance(inputs, torch.Tensor):
        inputs = torch.tensor(inputs, dtype=torch.float32)
    
    # Perform inference
    with torch.no_grad():
        outputs = model(inputs)
    
    # Convert outputs to labels using idx2str
    output_labels = [idx2str[idx] for idx in outputs.argmax(dim=1).tolist()]
    return output_labels

def test(model, dataset):
    print(f"Testing the model on the test set")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    Actual_Label, Predicted_Label = [], []
    for inputs, label in dataset:
        Actual_Label.append(label)

        inputs = torch.tensor(inputs, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, label)
        
        loss.backward()
        optimizer.step()

        Predicted_Label.append(outputs.argmax().item())

    print(f"Accuracy: {accuracy_score(Actual_Label, Predicted_Label) * 100}")
    print(f"Precision: {precision_score(Actual_Label, Predicted_Label, average=f'macro', zero_division=0) * 100}")
    print(f"Recall: {recall_score(Actual_Label, Predicted_Label, average=f'macro', zero_division=0) * 100}")
    print(f"F1-score: {f1_score(Actual_Label, Predicted_Label, average=f'macro', zero_division=0) * 100}")

def write_model_params_to_txt(model, file_path, prepend_sentence):
    with open(file_path, 'a') as f:
        f.write(prepend_sentence + '\n')
        for name, param in model.named_parameters():
            if param.requires_grad:
                f.write(f"{name}: {param.data}\n")

def train(client_model, dataset, num_epochs=10, lr=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(client_model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for inputs, labels in dataset:
            inputs = torch.tensor(inputs, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
            
            optimizer.zero_grad()
            
            outputs = client_model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
    return client_model