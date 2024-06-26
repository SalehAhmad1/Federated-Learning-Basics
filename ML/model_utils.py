import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

def test(model, datasets):
    print(f"Testing the model on the test set")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    Actual_Label, Predicted_Label = [], []
    for dataset in datasets:
        for data in dataset:
            inputs, labels = data
            
            inputs = torch.tensor(inputs, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            Actual_Label.append(labels.detach().numpy())
            Predicted_Label.append(outputs.argmax().detach().numpy())
    print(f"Accuracy: {accuracy_score(Actual_Label, Predicted_Label) * 100}")
    print(f"Precision: {precision_score(Actual_Label, Predicted_Label, average=f'macro') * 100}")
    print(f"Recall: {recall_score(Actual_Label, Predicted_Label, average=f'macro') * 100}")
    print(f"F1-score: {f1_score(Actual_Label, Predicted_Label, average=f'macro') * 100}")

def train(client_model, datasets, num_epochs=10, lr=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(client_model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for data in datasets:
            inputs, labels = data
            
            inputs = torch.tensor(inputs, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
            
            optimizer.zero_grad()
            
            outputs = client_model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
    return client_model