import json
import torch
from collections import Counter

class Utils:

    def __init__(self, vocab=None):
        self.vocab = vocab
        self.SOS_TOKEN = 0
        self.EOS_TOKEN = 1

    def get_device(self):
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        return device

    def load_json(self, filename):
        with open(filename, 'r') as file:
            return json.load(file)

    def tokenize_sentence(self, sentence):
        """Tokenize the text by converting to lowercase and splitting."""
        return sentence.lower().split()

    def build_vocab(self, data):
        """Build vocabulary from the provided data."""
        counter = Counter()
        for item in data:
            counter.update(self.tokenize_sentence(item['question']))
            counter.update(self.tokenize_sentence(item['answer']))

        self.vocab = {
            '<pad>': self.SOS_TOKEN,
            '<unk>': self.EOS_TOKEN
        }

        self.vocab.update({
            word: idx + 2
            for idx, (word, _) in enumerate(counter.most_common())
        })

        self.save_json_file(self.vocab, 'data/vocab.json')
        return self.vocab

    def save_json_file(self, data, file_path):
        """Save the vocabulary to a JSON file."""
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)
        print(f"File saved to {file_path}")

    def sentence_to_tokens(self, sentence, vocab, default='<pad>'):
        """Convert text to tensor using the vocabulary."""
        tokens = self.tokenize_sentence(sentence)
        token_indices = [vocab.get(token, vocab[default]) for token in tokens]
        if default == '<pad>':
            sos_index = vocab.get(default)
            return [sos_index] + token_indices
        else:
            eos_index = vocab.get(default)
            return token_indices + [eos_index]

    def build_qa_tokens(self, train_json_data, vocab):
        """Build QA tokens from the training data."""
        generated_data = []
        for item in train_json_data:
            question = item['question']
            answer = item['answer']
            qa_pair = {
                "question": self.sentence_to_tokens(question, vocab),
                "answer": self.sentence_to_tokens(answer, vocab, '<unk>')
            }
            generated_data.append(qa_pair)

        self.save_json_file(generated_data, 'data/qa_tokens.json')
        return generated_data



