import torch
import torch.nn as nn
import torch.optim as optim

from cryptography.fernet import Fernet
import io

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def average_weights(models):        
    avg_state_dict = {}
    for key in models[0].state_dict().keys():
        avg_state_dict[key] = torch.stack([model.state_dict()[key] for model in models], 0).mean(0)
    return avg_state_dict

def encrypt_weights_AES(model):
    key = Fernet.generate_key()
    cipher = Fernet(key)
    encrypted_weights = {}
    for name, param in model.named_parameters():
        buffer = io.BytesIO()
        torch.save(param.data, buffer)
        param_bytes = buffer.getvalue()
        encrypted_weights[name] = cipher.encrypt(param_bytes)
    return encrypted_weights, key

def decrypt_weights_AES(encrypted_weights, key):
    cipher = Fernet(key)
    decrypted_weights = {}
    for name, encrypted_params in encrypted_weights.items():
        decrypted_param_bytes = cipher.decrypt(encrypted_params)
        buffer = io.BytesIO(decrypted_param_bytes)
        param_data = torch.load(buffer)
        decrypted_weights[name] = param_data
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