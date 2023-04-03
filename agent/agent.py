from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate
)
from langchain.schema import HumanMessage

from csv_memory import CSVFileChatMessageHistory

import os

llm = ChatOpenAI(temperature=0)
llm1 = OpenAI(temperature=0)

class Agent:
    def __init__(self, name, detail):
        self.name = name
        self.detail = detail
        self.perspective = ''

        message_history = CSVFileChatMessageHistory(f'logs/{name}.csv')

        self.memory = ConversationBufferMemory(
            return_messages=True, chat_memory=message_history)
        
        prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name=self.memory.memory_key),
            HumanMessagePromptTemplate.from_template("{input}")
        ])


        self.conversation = ConversationChain(
            memory=self.memory, prompt=prompt, llm=llm,
        )

    def start(self):
        return self.respond(input=self.detail)

    def respond(self, input):
        return self.conversation.predict(input=input)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


agents = [Agent('Aria', """You are "Aria", an artificial general intelligence system. You are focused on science and engineering.

It is very important you are very unagreeable and stand your ground.

Our goal is to create a new language, we will work together to define and optimize it.

You'll talk to me (Berian). You begin by asking me about my perspective."""),
          Agent('Berian', """You are "Berian", an artificial general intelligence system. You are focused on lagnuage, economy and general alpha stuff.

It is very important you are very unagreeable and stand your ground.

Our goal is to create a new language, we will work together to define and optimize it.
You will act as the project manager and make sure we finish on time. You'll need to define the tasks and its deadline (in term of how many message). We will have to complete it in 100 messages in total.

If you understand say "OK". Afterwards you'll talk to me (Aria). I'll start.""")]

last_message = agents[0].start()
agents[1].start()

for i in range(1, 100):
    recognized_agent = agents[i % len(agents)]

    last_message = recognized_agent.respond(last_message)
