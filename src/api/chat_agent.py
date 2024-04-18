from agents.llama_index_chat_retrieval import get_new_agent

chat_agents = {}

class ChatSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        if session_id in chat_agents.keys():
            self.agent = chat_agents[session_id]
        else:
            self.agent = get_new_agent()
            chat_agents[session_id] = self.agent

    @classmethod
    def get_reponse(cls, session_id: str, question: str):
        response = cls(session_id).agent.chat(question)
        return {"response": response.response, "sources": response.sources}
    
    @classmethod
    def clear_session(cls, session_id: str):
        chat_agents.pop(session_id)