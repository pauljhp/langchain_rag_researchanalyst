from langchain.memory import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
import utils

STORE = {}

class ChatSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        if session_id not in STORE:
            STORE[session_id] = ChatMessageHistory()
        self._history = STORE[session_id]

    @property
    def session_history(
            self):
        return self._history
    
    @session_history.setter
    def session_history(self, new_messages: BaseChatMessageHistory):
        self._history.add_messages(new_messages.messages)

    @session_history.getter
    def session_history(
            self) -> BaseChatMessageHistory:
        return self._history