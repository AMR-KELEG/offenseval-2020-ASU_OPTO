from abc import ABC, abstractmethod
from utils import plot_confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score

class BaseModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def predict(self, df):
        pass

    @abstractmethod
    def save(self, **params):
        pass

    @abstractmethod
    def load(self, **params):
        '''params: a dictionary for the model's parameters '''
        pass

    def f1_score(self, df):
        pred = self.predict(df)
        return f1_score(y_true=df['target'], y_pred=pred, pos_label=1, average='binary')

    def accuracy(self, df):
        pred = self.predict(df)
        return (df['target'] == pred).mean()

    def precision(self, df):
        pred = self.predict(df)
        return precision_score(y_true=df['target'], y_pred=pred, pos_label=1)

    def recall(self, df):
        pred = self.predict(df)
        return recall_score(y_true=df['target'], y_pred=pred, pos_label=1)

    def f1_score(self, df):
        pred = self.predict(df)
        return f1_score(y_true=df['target'], y_pred=pred, pos_label=1)
    
    def report(self, df):
        prec = self.precision(df)
        recall = self.recall(df)
        accuracy = self.accuracy(df)
        f1 = self.f1_score(df)
        print('Accuracy {}\n F1 score {}\n"Precision/recall": {} / {}'.format(
            accuracy, f1, prec, recall))
        return None

    def plot_matrix(self, df):
        pred = self.predict(df)
        plot_confusion_matrix(y_true=df['target'], y_pred=pred, classes=[0, 1])
