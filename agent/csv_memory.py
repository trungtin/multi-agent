from typing import List
import os

from langchain.schema import (
    AIMessage,
    BaseChatMessageHistory,
    BaseMessage,
    HumanMessage,
    _message_to_dict,
    messages_from_dict,
    messages_to_dict,
)

from collections import namedtuple

fields = ('type', 'message')
ChatMessageHistory = namedtuple('ChatMessageHistory', fields)

class CSVFileChatMessageHistory(BaseChatMessageHistory):
    """Chat message history that stores history in a csv file.
    This class expects that path of a csv file is presented.

    Args:
        file_path: path of the csv file
    """

    def __init__(self, file_path: str):
        if not os.path.isabs(file_path):
            file_path = os.path.join(os.getcwd(), file_path)

        self.file_path = file_path
        if not os.path.exists(self.file_path):
            # create directory if not exists
            if not os.path.exists(os.path.dirname(self.file_path)):
                os.makedirs(os.path.dirname(self.file_path))

        # write new file on each initialization
        with open(self.file_path, "w") as f:
            f.write(','.join(fields) + '\n')

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        """Retrieve the messages from csv file"""
        import csv

        messages = []
        with open(self.file_path, "r") as f:
            f.readline() # Skip the header
            reader = csv.reader(f)
            for row in map(ChatMessageHistory._make, reader):
                if row.type == 'human':
                    messages.append(HumanMessage(content=row.message))
                elif row.type == 'ai':
                    messages.append(AIMessage(content=row.message))

        return messages

    def add_user_message(self, message: str) -> None:
        self.append(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        self.append(AIMessage(content=message))

    def append(self, message: BaseMessage) -> None:
        """Append the message to the record in DynamoDB"""
        import csv

        messages = messages_to_dict(self.messages)
        _message = _message_to_dict(message)
        messages.append(_message)

        with open(self.file_path, "a") as f:
            writer = csv.writer(f)
            writer.writerow([message.type, message.content])
    
    def clear(self):
        pass
