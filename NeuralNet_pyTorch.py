import pandas as pd

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

from tqdm import tqdm

import matplotlib.pyplot as plt



# NN class
class MultiHeadBinaryModel(nn.Module):
    def __init__(self):
        super(MultiHeadBinaryModel, self).__init__()
        self.fc1 = nn.Linear(7, 64)  # 42 / 104 is the number of features
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc5 = nn.Linear(256, 64)


        # we will treat each head as a binary classifier ...
        # ... so the output features will be 1
        # ... these final layers take the same inputs and output a single output.
        self.out1 = nn.Linear(64, 1)
        self.out2 = nn.Linear(64, 1)
        self.out3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(self.bn1(x))
        x = self.fc2(x)
        x = F.relu(self.bn2(x))
        x = self.fc3(x)
        x = F.relu(self.bn3(x))
        x = self.fc4(x)
        x = F.relu(self.bn4(x))
        x = self.fc5(x)

        # each binary classifier head will have its own output
        out1 = torch.sigmoid(self.out1(x))
        out2 = torch.sigmoid(self.out2(x))
        out3 = torch.sigmoid(self.out3(x))

        return out1, out2, out3

# `BinaryDataset()` class for multi-head binary classification model
# adaptable for our problem, with multitask binary classification
class BinaryDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        features = self.x[index, :]
        labels = self.y[index, :]

        # we have 12 feature columns
        features = torch.tensor(features, dtype=torch.float32)
        # there are 5 classes and each class can have a binary value ...
        # ... either 0 or 1
        label1 = torch.tensor(labels[0], dtype=torch.float32)
        label2 = torch.tensor(labels[1], dtype=torch.float32)
        label3 = torch.tensor(labels[2], dtype=torch.float32)
        #label4 = torch.tensor(labels[3], dtype=torch.float32)
        #label5 = torch.tensor(labels[4], dtype=torch.float32)
        return {
            'features': features,
            'label1': label1,
            'label2': label2,
            'label3': label3}






# loadin data set
def load_health_dataset(test_size = 0.3):
    health_data = pd.read_csv('data/simulated_df_27.07.21.csv', sep=',').drop('Unnamed: 0', axis = 1)
    health_data.head()

    targets_sel = ['FIBR_PREDS', 'ZSN', 'LET_IS']

    predictors_sel_7 = ['NITR_S',
                        'K_SH_POST',
                        'zab_leg_01',
                        'ZSN_A',
                        'n_r_ecg_p_05', 'AGE', 'SEX']



    y = health_data[targets_sel].values
    X = health_data[predictors_sel_7].values

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 15)

    return x_train, y_train, x_test, y_test

# loss function
def binary_loss_fn(outputs, targets):
    o1, o2, o3 = outputs
    t1, t2, t3 = targets
    l1 = nn.BCELoss()(o1.squeeze(), t1)
    l2 = nn.BCELoss()(o2.squeeze(), t2)
    l3 = nn.BCELoss()(o3.squeeze(), t3)
    return (l1 + l2 + l3) / 3         # return the average loss (adapt for 3 classes)


# training function
def train(model, dataloader, optimizer, binary_loss_fn, train_dataset, device):
    model.train()
    counter = 0
    train_running_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_dataset) / dataloader.batch_size)):
        counter += 1

        # extract the features and labels
        features = data['features'].to(device)
        target1 = data['label1'].to(device)
        target2 = data['label2'].to(device)
        target3 = data['label3'].to(device)

        # zero-out the optimizer gradients
        optimizer.zero_grad()

        outputs = model(features)
        targets = (target1, target2, target3)#, target4, target5)
        loss = binary_loss_fn(outputs, targets)
        train_running_loss += loss.item()

        # backpropagation
        loss.backward()
        # update optimizer parameters
        optimizer.step()

    train_loss = train_running_loss / counter
    return train_loss





def train_test_split_health_data():
    # prepare the dataset
    # x_train, y_train, _, _ = make_dataset(10000, 12, 5)
    x_train, y_train, x_test, y_test = load_health_dataset()
    # print some info
    print(f"[INFO]: Number of training samples: {x_train.shape[0]}")
    print(f"[INFO]: Number of training features: {x_train.shape[1]}")
    # train dataset
    train_dataset = BinaryDataset(x_train, y_train)
    test_dataset = BinaryDataset(x_test, y_test)

    return train_dataset, test_dataset

def training_network(train_dataset):


    # train data loader
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=1024)
    # initialize the model
    model = MultiHeadBinaryModel()


    # learning parameters
    optimizer = optim.Adam(params=model.parameters(), lr=0.001)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 100
    # load the model on to the computation device
    model.to(device)


    # start the training
    train_loss = []
    for epoch in range(epochs):
        #print(f"Epoch {epoch+1} of {epochs}")
        train_epoch_loss = train(
            model, train_dataloader, optimizer, binary_loss_fn, train_dataset, device
        )
        train_loss.append(train_epoch_loss)
        #print(f"Train Loss: {train_epoch_loss:.4f}")
    torch.save(model.state_dict(), 'pyTorch/outputs/multi_head_binary.pth')


    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('pyTorch/outputs/multi_head_binary_loss.png')
    plt.show()

def evaluating_network(test_dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1)


    # prepare the trained model
    model = MultiHeadBinaryModel()
    model.load_state_dict(torch.load('pyTorch/outputs/multi_head_binary.pth'))
    model.to(device)
    model.eval()

    all_outputs_list = []
    y_pred = []
    y = []

    y_pred_1 = []   #'FIBR_PREDS',
    y_pred_2 = []   #          'ZSN',
    y_pred_3 = []   #         'LET_IS'
    y_pred_1_prob = []   #'FIBR_PREDS',
    y_pred_2_prob = []   #          'ZSN',
    y_pred_3_prob = []   #         'LET_IS'
    y_1 = []        #'FIBR_PREDS',
    y_2 = []        #          'ZSN',
    y_3 = []        #         'LET_IS'


    for i, test_sample in enumerate(test_dataloader):
        #print(f"SAMPLE {i}")
        # extract the features and labels
        features = test_sample['features'].to(device)
        target1 = test_sample['label1'].to(device)
        target2 = test_sample['label2'].to(device)
        target3 = test_sample['label3'].to(device)

        outputs = model(features)           ## probabilities p: 0 < p < 1
        all_outputs_list.append(outputs)    # append model outputs (y_pred to model)


        # get all the labels
        all_labels = []
        for out in outputs:
            if out >= 0.5:
                all_labels.append(1)        ## binary decision
            else:
                all_labels.append(0)


        # list of lists with 333 x 3 components
        y_pred.append(all_labels)   ## prediciton of the model with 0 or 1

        y_pred_1.append(all_labels[0])
        y_pred_2.append(all_labels[1])
        y_pred_3.append(all_labels[2])

        y_pred_1_prob.append(outputs[0])
        y_pred_2_prob.append(outputs[1])
        y_pred_3_prob.append(outputs[2])

        #print('accuracy of the model on test set : ', (((outputs.reshape(-1).round() == outputs).sum()) / float(outputs.shape[0])).item(), "%")

        targets = (target1, target2, target3)

        # get all the targets in int format from tensor format
        all_targets = []
        for target in targets:
            all_targets.append(int(target.squeeze(0).detach().cpu()))

        # list of lists with 333 x 3 components
        y.append(all_targets)   ## actual value of the target data with 0 or 1
        y_1.append(all_targets[0])  # 'FIBR_PREDS',
        y_2.append(all_targets[1])  # 'ZSN',
        y_3.append(all_targets[2])  # 'LET_IS'
    #          'FIBR_PREDS',
     #          'ZSN',
      #         'LET_IS'

    auc_score = roc_auc_score(y, y_pred)

    auc_score_1 = roc_auc_score(y_1, y_pred_1)
    auc_score_2 = roc_auc_score(y_2, y_pred_2)
    auc_score_3 = roc_auc_score(y_3, y_pred_3)

    recall = recall_score(y, y_pred, average=None)
    precision = precision_score(y, y_pred, average=None)

    print(f"AUC-Score of pyTorch Model is: {auc_score}")

    print(f"AUC-Score of FIBR_PREDS (1) is: {auc_score_1}")
    print(f"AUC-Score of ZSN (2)is: {auc_score_2}")
    print(f"AUC-Score of LET_IS (3) is: {auc_score_3}")

    print(f"recall_score of pyTorch Model is: {recall}")
    print(f"precision_score of pyTorch Model is: {precision}")

    from sklearn.metrics import average_precision_score
    average_precision = average_precision_score(y, y_pred)

    print('Average precision-recall score: {0:0.2f}'.format(
        average_precision))

    #from sklearn.metrics import precision_recall_curve
    #from sklearn.metrics import plot_precision_recall_curve
    #import matplotlib.pyplot as plt

    #disp = plot_precision_recall_curve(model, x_test, y_test)
    #disp.ax_.set_title('2-class Precision-Recall curve: '
    #                  'AP={0:0.2f}'.format(average_precision))








if __name__ == '__main__':
    train_dataset, test_dataset = train_test_split_health_data()
    training_network(train_dataset)
    evaluating_network(test_dataset)



# score w/o batchnorm and dropout: AUC-Score of pyTorch Model is: 0.8431093361616138
# score after implementation of dropout/batchnorm: AUC-Score of pyTorch Model is: 0.8433316251052858
