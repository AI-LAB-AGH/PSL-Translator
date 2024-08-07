import json
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

def train(model: torch.nn.Module,
          train_loader: torch.utils.data.DataLoader,
          test_loader: torch.utils.data.DataLoader,
          num_epochs=10,
          lr = 0.001,
          crit=torch.nn.CrossEntropyLoss,
          optim=torch.optim.Adam,
          save_path=None) -> dict:
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = crit()
    optimizer = optim(model.parameters(), lr)
    history = {'epoch': [], 'loss': [], 'accuracy': []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            # Input dims: N x L x IN
            (lefts, rights), labels = data

            optimizer.zero_grad()
            outputs = None
            model.initialize_cell_and_hidden_state()
            for frame in range(len(lefts)):
                left = lefts[frame].unsqueeze(1)
                right = rights[frame].unsqueeze(1)
                outputs = model(left, right)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print(f'\rEpoch {epoch+1}/{num_epochs}, Batch {i+1}/{len(train_loader)} complete', end='')

        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for data in test_loader:
                (lefts, rights), labels = data

                outputs = None
                model.initialize_cell_and_hidden_state()
                for frame in range(len(lefts)):
                    left = lefts[frame].unsqueeze(1)
                    right = rights[frame].unsqueeze(1)
                    outputs = model(left, right)

                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        history['epoch'].append(epoch + 1)
        history['loss'].append(running_loss / len(train_loader))
        history['accuracy'].append(accuracy)

        print(f'\rEpoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}, Accuracy: {accuracy}')

    with open('training_history.json', 'w') as f:
        json.dump(history, f)
    
    if save_path is not None:
        torch.save(model.state_dict(), save_path)

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in test_loader:
            (lefts, rights), labels = data
            
            outputs = None
            model.initialize_cell_and_hidden_state()
            for frame in range(len(lefts)):
                left = lefts[frame].unsqueeze(1)
                right = rights[frame].unsqueeze(1)
                outputs = model(left, right)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    return {'history': history, 'accuracy': accuracy, 'cm': cm}


def display_results(results: dict, actions):
    history = results['history']
    accuracy = results['accuracy']
    cm = results['cm']

    print(f'Accuracy: {accuracy}')
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=actions, yticklabels=actions)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    epochs = history['epoch']
    loss = history['loss']
    accuracy = history['accuracy']

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy over Epochs')
    plt.legend()

    plt.show()