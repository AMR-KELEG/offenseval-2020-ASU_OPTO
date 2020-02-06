# Check vocab
# Use CV
import os
import time
import json
from pathlib import Path
import numpy as np
import torch
from torch import tensor
from torch import argmax, cat
from scipy.special import softmax
from models.basemodel import BaseModel
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from sklearn.metrics import f1_score, accuracy_score

class CustomDataset(Dataset):
    def __init__(self, dataframe, model_name, max_seq_length=75, pad_to_max_length=True):
        self.len = len(dataframe)
        self.data = dataframe
        self.max_seq_length = max_seq_length
        self.pad_to_max_length = pad_to_max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode(self, sentence):
        # This adds [CLS] and [SEP] by default
        return self.tokenizer.encode(
            sentence, max_length=self.max_seq_length, pad_to_max_length=self.pad_to_max_length)

    def __getitem__(self, index):
        item = self.data.iloc[index]
        text = item.text
        label = item.target
        X = self.encode(text)
        return tensor(X), label

    def __len__(self):
        return self.len

class BertModel(BaseModel):
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def get_params(self):
        return self.model.parameters()

    def reload(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).cuda()

    def fit(self, train_df, dev_df=None, epochs=4, learning_rate=1e-05, weight_decay=1e-5, params=None):
        if not params:
            params = {'batch_size': 20,
                     'shuffle': True,
                     'drop_last': False,
                     'num_workers': 1}
        train_dataloader = DataLoader(
            CustomDataset(train_df, self.model_name, max_seq_length=75, pad_to_max_length=True),
            **params)

        if dev_df is not None:
            test_dataloader = DataLoader(CustomDataset(dev_df,
                self.model_name, max_seq_length=75, pad_to_max_length=True),
                **params)

        # Create optimizer
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(params = self.get_params(), lr=learning_rate, weight_decay=weight_decay)

        train_losses = []
        train_f1 = []
        dev_losses = []
        dev_f1 = []

        # Loop over dataset
        self.model.train()
        for epoch in range(epochs):
            for batch_number, (tokens, labels) in enumerate(train_dataloader):
                tokens = tokens.cuda()
                labels = labels.cuda()
                optimizer.zero_grad()
                # TODO: The model returns a tuple, why?
                outputs = self.model.forward(tokens)[0]
                predicted = argmax(outputs, dim=1)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
            print('\n\nEpoch {} stats: '.format(epoch))
            print('Train')
            losses = []
            TP, TN, FP, FN = [], [], [], []
            for batch_number, (tokens, labels) in enumerate(train_dataloader):
                batch_metrics = self.compute_metrics(tokens, labels)
                losses.append(batch_metrics['loss'])
                TP.append(batch_metrics['TP'])
                TN.append(batch_metrics['TN'])
                FP.append(batch_metrics['FP'])
                FN.append(batch_metrics['FN'])

            print('Average Loss', sum(losses) / len(train_dataloader))

            TP = sum(sum(TP, []))
            TN = sum(sum(TN, []))
            FP = sum(sum(FP, []))
            FN = sum(sum(FN, []))

            precision_1 = 1.0 * TP / (TP+FP) if TP+FP else 0
            recall_1 = 1.0 * TP / (TP+FN) if TP+FN else 0
            f1_1 = (precision_1 * recall_1 * 2) / (precision_1 + recall_1) if precision_1 + recall_1 else 0
            
            precision_0 = 1.0 * TN / (TN+FN) if TN+FN else 0
            recall_0 = 1.0 * TN / (TN+FP) if TN+FP else 0
            f1_0 = (precision_0 * recall_0 * 2) / (precision_0 + recall_0) if precision_0 + recall_0 else 0
            f1 = (f1_0 + f1_1)/2
            print('F1 score', f1)
            train_losses.append(sum(losses) / len(train_dataloader))
            train_f1.append(f1)

            print('\n\nDev')
            if dev_df is not None:
                losses = []
                TP, TN, FP, FN = [], [], [], []
                for tokens,labels in test_dataloader:
                    batch_metrics = self.compute_metrics(tokens, labels)
                    losses.append(batch_metrics['loss'])
                    TP.append(batch_metrics['TP'])
                    TN.append(batch_metrics['TN'])
                    FP.append(batch_metrics['FP'])
                    FN.append(batch_metrics['FN'])
                print('Average Loss', sum(losses) / len(test_dataloader))

                TP = sum(sum(TP, []))
                TN = sum(sum(TN, []))
                FP = sum(sum(FP, []))
                FN = sum(sum(FN, []))

                precision_1 = 1.0 * TP / (TP+FP) if TP+FP else 0
                recall_1 = 1.0 * TP / (TP+FN) if TP+FN else 0
                f1_1 = (precision_1 * recall_1 * 2) / (precision_1 + recall_1) if precision_1 + recall_1 else 0

                precision_0 = 1.0 * TN / (TN+FN) if TN+FN else 0
                recall_0 = 1.0 * TN / (TN+FP) if TN+FP else 0
                f1_0 = (precision_0 * recall_0 * 2) / (precision_0 + recall_0) if precision_0 + recall_0 else 0
                f1 = (f1_0 + f1_1)/2
                print('F1 score', f1)
                dev_losses.append(sum(losses) / len(test_dataloader))
                dev_f1.append(f1)

        return {'macro_train_f1': train_f1,
                'train_loss': train_losses,
                'macro_dev_f1': dev_f1,
                'dev_loss': dev_losses}

    def save(self, **params):
        base_dir = params.get('base_dir', 'results')
        model_name = params.get('model_name', self.model_name)
        timestamp = str(int(time.time()))
        results = params.get('results', {})
        epochs = params.get('epochs', 'NA')
        learning_rate = params.get('learning_rate', 'NA')
        weight_decay = params.get('weight_decay', 'NA')
        comment = params.get('comment', '')

        # Make a directory in the form
        save_dir = Path(base_dir, timestamp)
        os.makedirs(save_dir)

        # Update the results
        results['save_timestamp'] = timestamp
        results['epochs'] = epochs
        results['learning_rate'] = learning_rate
        results['weight_decay'] = weight_decay
        results['comment'] = comment

        # Save the model and the results
        torch.save(self.model.state_dict(), str(Path(save_dir, model_name)))
        with open(str(Path(save_dir, 'results.json')), 'w') as f:
            json.dump(results, f)

        # TODO: Prepare the error report (save the error dataframe)

    def load(self, **params):
        model_name = params.get('model_name', None)
        self.model.load_state_dict(model_name)

    def predict(self, df, params=None):
        if not params:
            params = {'batch_size': 20,
             'shuffle': False,
             'drop_last': False,
             'num_workers': 1}

        dataloader = DataLoader(
            CustomDataset(df, self.model_name, max_seq_length=75, pad_to_max_length=True),
            **params)
        predictions = []
        self.model.eval()
        for tokens, labels in dataloader:
            tokens = tokens.cuda()
            labels = labels.cuda()
            outputs = self.model.forward(tokens)[0]
            predictions.append(argmax(outputs, dim=1))
        return cat(predictions, dim=0).cpu().tolist()

    def predict_prob(self, df):
      self.model.eval()

      return df['text'].apply(lambda sentence: softmax(self.model.forward(
            torch.tensor([self.tokenizer.encode(
            sentence, max_length=75, pad_to_max_length=True)]).cuda())[0][0].cpu().tolist())[1])

    def compute_classification_error(self, df):
        return abs(df['target'] - self.predict_prob(df))

    def sample_worst_errors(self, df, sample_size=50):
        df['classification_error'] = self.compute_classification_error(df)
        return df.sort_values(by='classification_error', ascending=False).head(sample_size)

    def compute_metrics(self, tokens, labels):
        # TODO: Fix the loss function averaging
        self.model.eval()
        loss_function = nn.CrossEntropyLoss()
        batch_size = labels.reshape(-1).size()[0]

        tokens = tokens.cuda()
        labels = labels.cuda()
        outputs = self.model.forward(tokens)[0]
        # Compute predictions
        predictions = argmax(outputs, dim=1).cpu()
        # Compute losses
        # TODO: Don't I need to average this???
        losses = loss_function(outputs, labels) * batch_size
        labels = labels.cpu()
        TP = (predictions == labels) & (predictions == 1)
        TN = (predictions == labels) & (predictions == 0)
        FP = (predictions != labels) & (predictions == 1)
        FN = (predictions != labels) & (predictions == 0)
        self.model.train()
        return {'loss': losses.item(),
                'TP': TP.tolist(),
                'TN': TN.tolist(),
                'FP': FP.tolist(),
                'FN': FN.tolist()}

if __name__ == '__main__':
    model = BertModel('bert-base-cased')
    print(model.model)
