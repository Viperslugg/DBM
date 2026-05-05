# Load important libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import nltk
import json
import random

nltk.download('punkt_tab')
nltk.download('wordnet')

class ChatModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(ChatModel, self).__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    # Input -> FC layer -> ReLU -> Dropout -> 2nd FC layer -> ReLU -> Dropout -> 3rd FC layer to output size
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

class Assistant:

    def __init__(self, intent_path, function_mappings=None):
        self.model = None
        self.intent_path = intent_path

        self.documents = []
        self.vocabulary = []
        self.intents = []
        self.intents_responses = {}

        self.function_mappings = function_mappings
        self.X = None
        self.y = None

    # Add helper function
    @staticmethod
    def tokenize_and_lemmatize(text):
        lemmatizer = nltk.WordNetLemmatizer()

        words = nltk.word_tokenize(text)
        words = [lemmatizer.lemmatize(word.lower()) for word in words]
        return words

    # Extract features from text input by labelling each word as 0 and 1 if recognized
    def bag_of_words(self, words):
        return [1 if word in words else 0 for word in self.vocabulary]

    # Parsing of JSON data
    def parse_intents(self):
        lemmatizer = nltk.WordNetLemmatizer()

        with open(self.intent_path, 'r') as f:
            intents_data = json.load(f)

        for intent in intents_data['intents']:
            if intent['tag'] not in self.intents:
                self.intents.append(intent['tag'])
                self.intents_responses[intent['tag']] = intent['responses']

            for pattern in intent['patterns']:
                pattern_words = self.tokenize_and_lemmatize(pattern)
                for w in pattern_words:
                    if w not in self.vocabulary:
                        self.vocabulary.append(w)
                self.documents.append((pattern_words, intent['tag']))
            self.vocabulary = sorted(set(self.vocabulary))

    def prepare_data(self):
        bags = []
        indices = []

        for documents in self.documents:
            words = documents[0]
            bag = self.bag_of_words(words)

            intent_index = self.intents.index(documents[1])

            bags.append(bag)
            indices.append(intent_index)

        self.X = np.array(bags)
        self.y = np.array(indices)

    # Training procedure
    def train_model(self, batch_size, lr, epochs):
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long)

        # Create dataset based on our tensors
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = ChatModel(self.X.shape[1], len(self.intents))

        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            running_loss = 0.0

            for batch_X, batch_y in loader:
                # Reset the gradient computation
                optimizer.zero_grad()
                # Predict which tag the model predicts
                prediction = self.model(batch_X)
                # Compute the loss between predicted and ground truths
                loss = loss_fn(prediction, batch_y)
                # Backpropagate the loss through the network
                loss.backward()
                # Take a step in the correct direction
                optimizer.step()
                running_loss += loss

            print(f"Epoch {epoch+1}: Loss is {running_loss/len(loader)}")

    def save_model(self, model_path, dimensions_path):
        # Store the learnable model parameters in an internal state dictionary
        torch.save(self.model.state_dict(), model_path)

        # Save the input and output sizes for creating an instance of the same model to a JSON file
        with open(dimensions_path, 'w') as f:
            json.dump({'input_size': self.X.shape[1],
                       'output_size': len(self.intents)}, f)

    def load_model(self, model_path, dimensions_path):
        with open(dimensions_path, 'r') as f:
            dimensions = json.load(f)

        # To load model weights, create an instance of the same model
        self.model = ChatModel(dimensions['input_size'],
                               dimensions['output_size'])
        # Load the parameters using the "load_state_dict" method. Set "weights_only = True"
        self.model.load_state_dict(torch.load(model_path, weights_only = True))

    def process_message(self, input_message):
        # Apply the same preprocessing procedure to the input message
        words = self.tokenize_and_lemmatize(input_message)
        bag = self.bag_of_words(words)

        bag_tensor = torch.tensor([bag], dtype=torch.float32)

        # Set the model for testing, no computation of gradients here
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(bag_tensor)

        # Take the largest probability score class and its value:
        probs = torch.softmax(predictions, dim=1)
        max_p, predicted_intent = torch.max(probs, 1)

        max_p = max_p.item()
        predicted_intent_index = predicted_intent.item()
        predicted_intent = self.intents[predicted_intent_index]

        ######### Uncomment if self.function_mappings is not empty ###########
        # if self.function_mappings:
        #    if predicted_intent in self.function_mappings:
        #        self.function_mappings[predicted_intent]()

        # The confidence threshold set here is subjective and should depend on factors
        # Here, 0.5 will be used as it is "neutral"; the median
        if self.intents_responses[predicted_intent] and max_p > 0.50:
            return random.choice(self.intents_responses[predicted_intent])
        else:
            # Print the output message "Keine Ahnung!", which translates to "No Idea!"
            string = "Keine Ahnung!"
            return string

if __name__ == '__main__':
    assistant = Assistant('german.json', function_mappings = None)
    assistant.parse_intents()
    assistant.load_model('chatbot_model.pth', 'dimensions.json')

    while True:
        message = input('Enter your message (Type q to quit):')

        if message == 'q':
            break

        print(assistant.process_message(message))

