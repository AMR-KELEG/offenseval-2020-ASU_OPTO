from models.basemodel import BaseModel

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

class tfidfModel(BaseModel):
    def __init__(self):
        self.pipeline = make_pipeline(TfidfVectorizer(ngram_range=(1, 3), analyzer='word'), LogisticRegression(random_state=42))
    
    def fit(self, train_df):
        self.pipeline.fit(train_df['text'], train_df['target'])
    
    def predict(self, df):
        return self.pipeline.predict(df['text'])
    
    def save(self):
        # TODO
        pass
    
    def load(self, model_params):
        # TODO
        pass
