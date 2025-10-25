from django.shortcuts import render
from django.http import JsonResponse
from chats.models import ChatMessage
from .replies import bot_reply

def chatbot(request):
    
    if not request.session.session_key:
        request.session.create()

    session_key = request.session.session_key
    messages = ChatMessage.objects.filter(session_key=session_key).order_by('timestamp')

    return render(request, 'chatbot.html', {'messages': messages})

def chat_api(request):
    if request.method == 'POST':
        session_key = request.session.session_key
        user_msg = request.POST.get('message')
        ChatMessage.objects.create(session_key=session_key, sender='user', message=user_msg)
        reply = bot_reply(user_msg)
        ChatMessage.objects.create(session_key=session_key, sender='bot', message=reply)

        return JsonResponse({'reply': reply})

