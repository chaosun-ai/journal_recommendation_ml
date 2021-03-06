import numpy as np
from tensorflow.keras.models import load_model
from joblib import load

def journal_pred(paper_title, key_words):

    input_text = [paper_title + key_words]
    vectorizer = load('vectorizer.joblib')
    input_x = vectorizer.transform(input_text)
    input_x = input_x.todense()
    
    model = load_model('journal_mlp_model.h5')
    y_pred = model.predict(input_x)
    y_pred = np.squeeze(y_pred)
    label = np.argpartition(y_pred, -10)[-10:]
    #print(label)
    label_encoder = load('le.joblib')
    journal_list = label_encoder.inverse_transform(label)
    print(journal_pred)

    return journal_list


if __name__ == '__main__':
    
    paper_title = 'Machine Learning for engineering journal selection'
    key_words = 'machine learning engineering recommendation system'

    journal_list = journal_pred(paper_title, key_words)
    print(journal_list)
