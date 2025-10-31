from django.shortcuts import render
from django.http import JsonResponse
from chats.models import ChatMessage
from django.views.decorators.csrf import csrf_exempt
import os, json, torch, random, nltk, numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTENTS_PATH = os.path.join(BASE_DIR, "intents.json")


class ChatbotModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


class ChatbotAssistant:
    def __init__(self, intents_path):
        self.intents_path = intents_path
        self.model = None
        self.vocab = []
        self.intents = []
        self.intents_responses = {}
        self.documents = []

    def tokenize(self, text):
        lemmatizer = nltk.WordNetLemmatizer()
        words = nltk.word_tokenize(text)
        return [lemmatizer.lemmatize(w.lower()) for w in words]

    def bag_of_words(self, words):
        return [1 if w in words else 0 for w in self.vocab]

    def parse_intents(self):
        with open(self.intents_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for intent in data["intents"]:
            if intent["tag"] not in self.intents:
                self.intents.append(intent["tag"])
                self.intents_responses[intent["tag"]] = intent["responses"]
            for pattern in intent["patterns"]:
                w = self.tokenize(pattern)
                self.vocab.extend(w)
                self.documents.append((w, intent["tag"]))
        self.vocab = sorted(set(self.vocab))

    def prepare_data(self):
        X, y = [], []
        for words, tag in self.documents:
            X.append(self.bag_of_words(words))
            y.append(self.intents.index(tag))
        return np.array(X), np.array(y)

    def load_model(self, model_path, dim_path):
        with open(dim_path, "r") as f:
            dims = json.load(f)
        self.model = ChatbotModel(dims["input_size"], dims["output_size"])
        self.model.load_state_dict(torch.load(model_path, weights_only=True))

    def process_message(self, text):
        words = self.tokenize(text)
        bag = torch.tensor([self.bag_of_words(words)], dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(bag)
        intent_idx = torch.argmax(outputs, dim=1).item()
        intent = self.intents[intent_idx]
        return random.choice(self.intents_responses.get(intent, ["Sorry, I didn’t understand."]))
    
def bot_reply(query):
    assistant = ChatbotAssistant(INTENTS_PATH)
    assistant.parse_intents()
    assistant.load_model("chatbot_model.pth", "dimensions.json")
    return assistant.process_message(query)



def chatbot(request):
    
    if not request.session.session_key:
        request.session.create()

    session_key = request.session.session_key
    messages = ChatMessage.objects.filter(session_key=session_key).order_by('timestamp')

    return render(request, 'chatbot.html', {'messages': messages})

@csrf_exempt
def chat_api(request):
    if request.method == 'POST':
        if not request.session.session_key:
            request.session.create()

        session_key = request.session.session_key
        user_msg = request.POST.get('message')

        ChatMessage.objects.create(session_key=session_key, sender='user', message=user_msg)

        reply = bot_reply(user_msg)

        ChatMessage.objects.create(session_key=session_key, sender='bot', message=reply)

        return JsonResponse({'reply': reply})
    
   
    return JsonResponse({'error': 'GET not allowed, use POST instead.'}, status=405)



