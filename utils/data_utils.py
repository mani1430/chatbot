import json
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import torch
from torch.utils.data import Dataset


def load_data(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file at path {file_path} does not exist.")

    with open(file_path, 'r') as file:
        data = json.load(file)

    return data


def preprocess_data(data):
    """Preprocess data for training."""
    questions = [item['question'] for item in data]
    answers = [item['answer'] for item in data]

    # Split data
    train_questions, val_questions, train_answers, val_answers = train_test_split(
        questions, answers, test_size=0.2, random_state=42
    )

    # Vectorizer
    vectorizer = CountVectorizer()
    vectorizer.fit(train_questions + train_answers)

    return train_questions, train_answers, val_questions, val_answers, vectorizer


class QADataset(Dataset):
    """Dataset for question-answer pairs."""

    def __init__(self, questions, answers, vectorizer):
        self.questions = questions
        self.answers = answers
        self.vectorizer = vectorizer

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.answers[idx]
        question_vector = torch.tensor(self.vectorizer.transform([question]).toarray(), dtype=torch.float32)
        answer_vector = torch.tensor(self.vectorizer.transform([answer]).toarray(), dtype=torch.float32)
        return question_vector, answer_vector
