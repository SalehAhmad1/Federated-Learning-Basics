import torch
import torch.nn as nn
import torch.optim as optim

def average_weights(models):
    avg_state_dict = {}
    for key in models[0].state_dict().keys():
        avg_state_dict[key] = torch.stack([model.state_dict()[key] for model in models], 0).mean(0)
    return avg_state_dict

def save_model(model, filepath='./ServerModel.pth'):
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to '{filepath}'")

def load_model(model_class, filepath='./ServerModel.pth'):
    model = model_class()
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

def train(server_model, client_models, datasets, num_epochs=10, lr=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(server_model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for client_model, dataset in zip(client_models, datasets):
            for data in dataset:
                inputs, labels = data
                
                inputs = torch.tensor(inputs, dtype=torch.float32)
                labels = torch.tensor(labels, dtype=torch.long)
                
                optimizer.zero_grad()
                
                outputs = client_model(inputs)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
        
        # Share and update server model
        server_model.load_state_dict(average_weights(client_models))