import sklearn
import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, recall_score

def get_sampler(dataset):
    # https://towardsdatascience.com/demystifying-pytorchs-weightedrandomsampler-by-example-a68aceccb452
    class_counts = dataset.labels.value_counts()
    sample_weights = [1/class_counts[i] for i in dataset.labels.values]
    sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weights, num_samples=len(dataset), replacement=True)
    return sampler

def get_metrics(val_dataloader, model, include_features=False):
    model.eval()

    with torch.no_grad():
        all_preds=[]
        all_labels=[]

        for data in val_dataloader:
            # get predicted probabilities, and labels with highest probability
            if include_features is True:
                predicted_labels = model(data['text'], data['features'])
            elif include_features == "only":
                predicted_labels = model(data['features'])
            else:
                predicted_labels = model(data['text'])

            #compute argmax
            _, preds = torch.max(predicted_labels, 1)

            all_preds.extend(preds)
            all_labels.extend(data['labels'])
 
    f1 = f1_score(y_true=all_labels, y_pred=all_preds)
    accuracy = accuracy_score(y_true=all_labels, y_pred=all_preds)
    recall = recall_score(y_true=all_labels, y_pred=all_preds)
    
    return f1, accuracy, recall


def train_model(model, train_dataloader, val_dataloader, optimizer, loss_function, epochs, include_features=False):
 # Sets the module in training mode.
    val_f1s=[]
    val_accuracies=[]
    val_recalls=[]

    for n_epoch in range(epochs):

        model.train()

        for data in train_dataloader:
            if include_features is True:
                y_hat = model(data['text'], data['features'])
            elif include_features == "only":
                y_hat = model(data['features'])
            else:
                y_hat = model(data['text'])

            optimizer.zero_grad()
            loss = loss_function(y_hat, data['labels'])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        f1, accuracy, recall = get_metrics(val_dataloader, model, include_features)
        val_f1s.append(f1)
        val_accuracies.append(accuracy)
        val_recalls.append(recall)
        
        if n_epoch % 100 == 0:
            print(f'At epoch {n_epoch}: loss = {loss:.3f}, and accuracy = {accuracy:.3f}')
    return val_f1s, val_accuracies, val_recalls



def plot_metrics(metrics):
    f1s, accuracies, recalls = metrics
    plt.plot(range(1, len(accuracies)+1), f1s, label='f1')
    plt.plot(range(1, len(accuracies)+1), accuracies, label='accuracy')
    plt.plot(range(1, len(accuracies)+1), recalls, label='recall')
    plt.legend()

    print(f"Final values: f1 {f1s[-1]:.3f}, accuracy {accuracies[-1]:.3f}, recall {recalls[-1]:.3f}")


#OTHER HELPERS 
def encode_output_variable(filepath, svm=None):
    df = pd.read_csv(filepath)
    if svm:
        df[['y']] = -1
    else:
        df[['y']] = 0
    df.loc[df.loc[:,'Overall Compliance'] == "No",'y'] = 1
    return df