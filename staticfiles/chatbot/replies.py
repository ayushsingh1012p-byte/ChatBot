from . import main

def bot_reply(query):
    chatbot = main.ChatbotAssistant(main.file_path)
    print(chatbot.tokenize_and_lemmatize('run running runs ran'))

# In[14]:


    assistant = main.ChatbotAssistant(main.file_path)
    assistant.parse_intents()
    assistant.prepare_data()
    assistant.train_model(batch_size=8, lr=0.001, epochs=100)
    assistant.save_model('chatbot_model.pth', 'dimensions.json')


    # In[6]:



    assistant = main.ChatbotAssistant(main.file_path)
    assistant.parse_intents()
    assistant.load_model('chatbot_model.pth', 'dimensions.json')
    return str(assistant.process_message(query))


