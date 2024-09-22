import torch
from models.decoder import Decoder
from models.encoder import Encoder
from utils.grammer import Grammer
from utils.utils import Utils
import time
import datetime

class BotPredict:
    def __init__(self):
        self.utils = Utils()
        self.vocab = self.utils.load_json('data/vocab.json')

    def prepare_data(self):
        device = self.utils.get_device()
        vocab_length = len(self.vocab)
        output_size = vocab_length + 1
        input_size = vocab_length
        hidden_size = 256
        encoder = Encoder(input_size, hidden_size).to(device)
        decoder = Decoder(hidden_size, output_size).to(device)
        return encoder, decoder

    def evaluate_with_scores(self, encoder, decoder, sentence):
        device = self.utils.get_device()

        with torch.no_grad():
            input_tensor = torch.tensor(sentence, dtype=torch.long).to(device)
            encoder_hidden = encoder.initHidden().to(device)

            # Encode the input
            for i in range(input_tensor.size(0)):
                encoder_output, encoder_hidden = encoder(input_tensor[i], encoder_hidden)

            # Decode the input into an output sequence
            decoder_input = torch.tensor([0]).to(device)  # Start token
            decoder_hidden = encoder_hidden

            decoded_words = []
            confidence_scores = []

            for i in range(len(self.vocab)):  # Max length of response
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_output.data.topk(1)
                confidence_score = torch.exp(topv).item()
                confidence_scores.append(confidence_score)

                if topi.item() == 1:  # Stop if token is predicted
                    break
                else:
                    decoded_words.append(topi.item())

                decoder_input = topi.squeeze(0).detach()  # Use predicted token as input for next time step

        return decoded_words, confidence_scores

    def token_to_sentence(self, tokens, vocab):
        reverse_vocab = {idx: word for word, idx in vocab.items()}
        words = []
        for token in tokens:
            words.append(reverse_vocab.get(token, ''))  # Handle unknown tokens
        sentence = ' '.join(words).strip()
        return sentence

    def validate_input_tokens(self, tokens):
        if all(token == 0 for token in tokens):
            print("I'm sorry, I didn't understand that.")
            return False
        return True

    def check_confidence_threshold(self, scores, predicted_output):
        confidence_threshold = 0.7
        uncertain_indices = [i for i, score in enumerate(scores) if score < confidence_threshold]

        if uncertain_indices:
            print("I'm sorry, I didn't understand that...")
            return False
        return True

    def predict(self, sentence):
        # grammer_correction = Grammer()
        # corrected_sentence =  grammer_correction.correct_grammer(sentence)
        input_token = self.utils.sentence_to_tokens(sentence, self.vocab)

        if not self.validate_input_tokens(input_token):
            return  # Stop execution if input is invalid

        encoder, decoder = self.prepare_data()
        encoder.load_state_dict(torch.load('saved/encoder_model.pth', weights_only=True))
        decoder.load_state_dict(torch.load('saved/decoder_model.pth', weights_only=True))
        predicted_output, scores = self.evaluate_with_scores(encoder, decoder, input_token)

        if not self.check_confidence_threshold(scores, predicted_output):
            return

        output = self.token_to_sentence(predicted_output, self.vocab)
        # print('Predicted Output: ', predicted_output)
        # print('Predicted Output Sentence:', output)
        print('Predicted Score: ', scores)
        if output:  # Check if output is not empty
            for word in output.split():
                for character in word:
                    print(character, end='', flush=True)
                    time.sleep(0.1)
                print(' ', end='', flush=True)
            print()
        else:
            print("Bot: I'm sorry, I didn't understand that.")

    def chat(self):
        print("Chat with the bot! Type 'exit' to end the conversation.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break
            print("Bot is thinking...")
            print("Bot:", end='', flush=True)
            time.sleep(1)  # Simulate thinking time
            self.predict(user_input)

# Instantiate and use the bot
bot = BotPredict()
# print(f"Today is {datetime.date.today()}.")
bot.chat()
