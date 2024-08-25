# predict.py

import torch
from utils.data_utils import load_data, preprocess_data
from models.qa_model import QAModel
import numpy as np


def predict(question):
    # Load model and data
    file_path = './data/seq2seq.json'
    data = load_data(file_path)
    _, _, _, _, vectorizer = preprocess_data(data)

    model = QAModel(len(vectorizer.get_feature_names_out()), 100, len(vectorizer.get_feature_names_out()))
    model.load_state_dict(torch.load('qa_model.pth', weights_only=True))
    model.eval()

    # Vectorize the input question
    question_vector = torch.tensor(vectorizer.transform([question]).toarray(), dtype=torch.float32)

    with torch.no_grad():
        output_vector = model(question_vector).numpy().flatten()

    # Get indices of top values (you might need to modify based on actual model output)
    output_indices = np.argsort(output_vector)[::-1]  # Sort in descending order of values

    # Convert output vector to text
    output_terms = vectorizer.get_feature_names_out()

    # Construct the output text
    # Example: Take top 5 terms
    top_indices = output_indices[:5]
    answer = ' '.join(output_terms[i] for i in top_indices if output_vector[i] > 0)

    return answer


if __name__ == '__main__':
    question = "What is the capital of india?"
    print(predict(question))
