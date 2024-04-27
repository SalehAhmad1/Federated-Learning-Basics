import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import numpy as np

def save_params_to_file(params, filepath):
    with open(filepath, 'w') as f:
        for key, value in params.items():
            f.write(f'{key}: {value.tolist()}\n')

def average_weights(models, save_path='./parameters/averaged_weights.txt'):
    avg_state_dict = {}
    for key in models[0].state_dict().keys():
        avg_state_dict[key] = torch.stack([model.state_dict()[key] for model in models], 0).mean(0)

    # Save averaged weights
    save_params_to_file(avg_state_dict, save_path)

    return avg_state_dict

def save_model(model, filepath='./model/server_model.pth'):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to '{filepath}'")

def load_model(model_class, args, filepath='./model/server_model.pth'):
    model = model_class(*args)
    model.load_state_dict(torch.load(filepath))
    model.eval()
    print(f"Model loaded from '{filepath}'")
    return model

def train(server_model, client_models, datasets, num_epochs=10, lr=0.01, save_path='./parameters/client_models'):
    print(f'Training Client Models')
    os.makedirs(save_path, exist_ok=True)

    for epoch in tqdm(range(num_epochs)):
        for i, (client_model, dataset) in enumerate(zip(client_models, datasets)):
            client_model.load_state_dict(server_model.state_dict())
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(client_model.parameters(), lr=lr)
            for data in dataset:
                inputs, labels = data

                inputs = torch.tensor(inputs, dtype=torch.float32)
                labels = torch.tensor(labels, dtype=torch.long)

                optimizer.zero_grad()

                outputs = client_model(inputs)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

            # Save client model parameters
            save_params_to_file(client_model.state_dict(), f'{save_path}/client_model_{i}.txt')

        # Share and update server model
        server_model.load_state_dict(average_weights(client_models))
    return server_model

def test(model, dataset):
    print(f'Testing 1 Client Model')

    Actuals, Predicteds = [], []
    for data in tqdm(dataset):
        inputs, labels = data

        inputs = torch.tensor(inputs, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        Actuals.append(labels.detach().numpy())

        outputs = model(inputs)
        Predicteds.append(outputs.argmax().detach().numpy())

    print(f"Accuracy: {accuracy_score(Actuals, Predicteds) * 100}")
    print(f"Precision: {precision_score(Actuals, Predicteds, average=f'macro') * 100}")
    print(f"Recall: {recall_score(Actuals, Predicteds, average=f'macro') * 100}")
    print(f"F1-score: {f1_score(Actuals, Predicteds, average=f'macro') * 100}")