import json
import torch
import random
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
    
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #model.to(device)
    criterion = crit()
    optimizer = optim(model.parameters(), lr)
    history = {'epoch': [], 'loss': [], 'accuracy': []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            # Input dims: N x L x IN
            inputs, labels = data

            optimizer.zero_grad()
            outputs = None
            model.initialize_cell_and_hidden_state()
            while outputs is None:
                cut = 0
                cut = int(len(inputs) * 0.1)
                for frame in range(cut, len(inputs) - cut):
                    skip = random.randint(0, 3)
                    if not skip:
                        x = inputs[frame]
                        outputs = model(x)

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
                inputs, labels = data

                optimizer.zero_grad()
                outputs = None
                model.initialize_cell_and_hidden_state()
                for frame in range(len(inputs)):
                    x = inputs[frame]
                    outputs = model(x)

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
            inputs, labels = data

            optimizer.zero_grad()
            outputs = None
            model.initialize_cell_and_hidden_state()
            for frame in range(len(inputs)):
                x = inputs[frame]
                outputs = model(x)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    return {'history': history, 'accuracy': accuracy, 'cm': cm}


def train_forecaster(model: torch.nn.Module,
          train_loader: torch.utils.data.DataLoader,
          test_loader: torch.utils.data.DataLoader,
          num_epochs=10,
          lr = 0.001,
          crit=torch.nn.MSELoss, 
          optim=torch.optim.Adam,
          save_path=None) -> dict:
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = crit()
    optimizer = optim(model.parameters(), lr)
    history = {'epoch': [], 'loss': [], 'mse': []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            # Input dims: N x L x IN
            inputs, _ = data
            
            optimizer.zero_grad()
            model.initialize_cell_and_hidden_state() 
            batch_loss = 0.0
            loss = 0.0
            
            for frame in range(len(inputs) - 1):
                x = inputs[frame]
                outputs = model(x)
                outputs = outputs.view(outputs.shape[1] // 2, 2)
                tgt = inputs[frame+1][0].float()
                temp = criterion(outputs, tgt)
                loss += temp
                batch_loss += temp.item()

            if type(loss) != float:
                loss.backward()
                optimizer.step()
                
            running_loss += batch_loss / (len(inputs) - 1)
            print(f'\rEpoch {epoch+1}/{num_epochs}, Batch {i+1}/{len(train_loader)} complete', end='')
            
        model.eval()
        mse_values = []
        with torch.no_grad():
            for data in test_loader:
                inputs, _ = data

                optimizer.zero_grad()
                outputs = None
                model.initialize_cell_and_hidden_state()
                for frame in range(len(inputs)-1):
                    x = inputs[frame]
                    outputs = model(x)
                    outputs = outputs.view(outputs.shape[1] // 2, 2)
                    mse = criterion(outputs, inputs[frame+1][0].float())
                    mse_values.append(mse.item())
        
        avg_mse = sum(mse_values) / len(mse_values)
        history['epoch'].append(epoch + 1)
        history['loss'].append(running_loss / len(train_loader))
        history['mse'].append(avg_mse)

        print(f'\rEpoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}, Avg MSE: {avg_mse}')

    with open('training_history.json', 'w') as f:
        json.dump(history, f)
    
    if save_path is not None:
        torch.save(model.state_dict(), save_path)

    model.eval()
    mse_values = []
    with torch.no_grad():
        for data in test_loader:
            inputs, _ = data

            optimizer.zero_grad()
            outputs = None
            model.initialize_cell_and_hidden_state()
            for frame in range(len(inputs)-1):
                x = inputs[frame]
                outputs = model(x)
                outputs = outputs.view(outputs.shape[1] // 2, 2)
                mse = criterion(outputs, inputs[frame+1][0].float())
                mse_values.append(mse.item())

    avg_mse = sum(mse_values) / len(mse_values)
    return {'history': history, 'avg_mse': avg_mse}


def display_results(results: dict, actions=None):
    history = results['history']
    
    if 'accuracy' in results:
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
    elif 'mse' in history:
        avg_mse = results['avg_mse']
        print(f'Average MSE: {avg_mse}')

        epochs = history['epoch']
        loss = history['loss']
        mse = history['mse']

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, loss, label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, mse, label='Validation MSE')
        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.title('Validation MSE over Epochs')
        plt.legend()

        plt.show()
    
