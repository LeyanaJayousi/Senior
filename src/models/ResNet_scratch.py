import csv
import os
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def Train(model, criterion, optimizer, num_epochs, batch_size, dataloaders, model_out_path, csv_out_path):
    """
    Method to train the model , with necessary parameters

    """
    model = model.to(device)

    highest_acc = 0.0
    epoch_history =[]

    for epoch in range(num_epochs):
        for phase in ['training', 'validation']:
            if phase == 'training':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            all_labels = []
            all_preds = []

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'training'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == 'training':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_accuracy = accuracy_score(all_labels, all_preds)
            epoch_balanced_accuracy = balanced_accuracy_score(
                all_labels, all_preds)

            print(f'{phase.capitalize()} Epoch {epoch + 1}/{num_epochs} '
                  f'Loss: {epoch_loss:.4f} Accuracy: {epoch_accuracy:.4f} '
                  f'Balanced Accuracy: {epoch_balanced_accuracy:.4f}\n')

            epoch_history.append([epoch + 1, phase, epoch_loss, epoch_accuracy, epoch_balanced_accuracy])

            if phase == 'validation' and epoch_accuracy > highest_acc:
                highest_acc = epoch_accuracy
        with open(csv_out_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if epoch == 0:
                writer.writerow(['Epoch', 'Phase', 'Loss', 'Accuracy', 'Balanced Accuracy'])
            writer.writerow([epoch + 1, phase, epoch_loss, epoch_accuracy, epoch_balanced_accuracy])


    torch.save(model.state_dict(), model_out_path)

    print('Highest accuracy is: {:.4f}'.format(highest_acc))
    return highest_acc