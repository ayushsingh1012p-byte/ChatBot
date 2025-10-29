from . import main

def bot_reply(query):
    assistant = main.ChatbotAssistant(main.INTENTS_PATH)
    assistant.parse_intents()
    assistant.load_model("chatbot_model.pth", "dimensions.json")
    return assistant.process_message(query)
