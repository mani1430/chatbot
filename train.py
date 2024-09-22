from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from models.decoder import Decoder
from models.encoder import Encoder
from utils.dataset import QADataset
from utils.utils import Utils


class BotTrain:
    def __init__(self, vocab=None):
        self.utils = Utils()
        self.data_file_path = 'data/new_data.json'
        self.multiple_question_answers_path = 'data/data.json'
        self.SOS_TOKEN = 0
        self.EOS_TOKEN = 1

    def load_data(self):
        json_data = self.utils.load_json(self.multiple_question_answers_path)
        mapped_data = []
        for entry in json_data:
            for question in entry["questions"]:
                mapped_data.append({
                    "question": question,
                    "answer": entry["answer"]
                })
        self.utils.save_json_file(mapped_data, self.data_file_path)

    def prepare_data(self):
        train_json_data = self.utils.load_json(self.data_file_path)
        vocab = self.utils.build_vocab(train_json_data)
        tokenized_data = self.utils.build_qa_tokens(train_json_data, vocab)
        dataset = QADataset(tokenized_data)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        vocab_length = len(vocab)
        output_size = vocab_length + 1
        input_size = vocab_length

        return input_size, output_size, dataloader

    def train(self, num_epochs):
        hidden_size = 256
        device = self.utils.get_device()
        input_size, output_size, dataloader = self.prepare_data()
        encoder = Encoder(input_size, hidden_size).to(device)
        decoder = Decoder(hidden_size, output_size).to(device)
        loss_fn = nn.NLLLoss()
        encoder_optimizer = optim.SGD(encoder.parameters(), lr=0.01)
        decoder_optimizer = optim.SGD(decoder.parameters(), lr=0.01)

        # Training loop
        for epoch in range(num_epochs):
            for input_tensor, target_tensor in dataloader:
                encoder_hidden = encoder.initHidden().to(device)
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()

                # Move tensors to device
                input_tensor, target_tensor = input_tensor.to(device), target_tensor.to(device)

                # Encode the input
                loss = 0
                for i in range(input_tensor.size(1)):
                    encoder_output, encoder_hidden = encoder(input_tensor[0][i], encoder_hidden)

                # Prepare the decoder input
                decoder_input = torch.tensor([self.SOS_TOKEN]).to(device)  # Start token
                decoder_hidden = encoder_hidden

                for i in range(target_tensor.size(1)):
                    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                    loss += loss_fn(decoder_output, target_tensor[0][i].unsqueeze(0))
                    decoder_input = target_tensor[0][i]  # Teacher forcing

                # Backpropagation and optimization
                loss.backward()
                encoder_optimizer.step()
                decoder_optimizer.step()

                print(f'Epoch {epoch + 1}, Loss: {loss.item() / target_tensor.size(1)}')

        # Save the trained models
        torch.save(encoder.state_dict(), 'saved/encoder_model.pth')
        torch.save(decoder.state_dict(), 'saved/decoder_model.pth')
        print("Models saved.")

# Instantiate and train the bot
bot = BotTrain()
# bot.load_data()
bot.train(50)
