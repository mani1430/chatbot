import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from utils.data_utils import load_data, preprocess_data, QADataset
from models.qa_model import QAModel
import torch.nn as nn

def train_model():
    # Load and preprocess data
    file_path = './data/seq2seq.json'
    data = load_data(file_path)
    train_questions, train_answers, val_questions, val_answers, vectorizer = preprocess_data(data)

    # Create datasets and dataloaders
    train_dataset = QADataset(train_questions, train_answers, vectorizer)
    val_dataset = QADataset(val_questions, val_answers, vectorizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Initialize the model, loss function, and optimizer
    input_dim = len(vectorizer.get_feature_names_out())
    hidden_dim = 100
    output_dim = len(vectorizer.get_feature_names_out())
    model = QAModel(input_dim, hidden_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 1000
    for epoch in range(num_epochs):
        model.train()
        for questions, answers in train_loader:
            optimizer.zero_grad()
            outputs = model(questions)
            loss = criterion(outputs, answers)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

    # Save the model
    torch.save(model.state_dict(), 'qa_model.pth')

if __name__ == '__main__':
    train_model()
