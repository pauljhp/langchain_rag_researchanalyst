from agents.llama_index_chat_retrieval import ChatAgent
from llama_index.core.agent import AgentChatResponse
from typing import Dict, Any, Literal, Optional


chat_agents = {}

def parse_response(response: AgentChatResponse) -> Dict[str, Any]:
    response_text = response.response
    source_nodes = response.source_nodes
    sources = response.sources
    accepted_fields = ["date", "id_bb_global", "ticker", "date", "source", 
                       "name", "title", "analyst"]
    parsed_sources, raw_contents, tools_used = [], [], []
    for source_node in source_nodes:
        # tools_used = 
        accepted_metadata = {k: v for k, v in source_node.metadata.items() if k.lower() in accepted_fields}
        parsed_sources.append(accepted_metadata)
    for source in sources:
        tool_name = source.dict().get("tool_name")
        tools_used.append(tool_name)
        content = source.content
        raw_contents.append(content)
    return {
        "response": response_text, 
        "sources": parsed_sources,
        "tools_used": tools_used,
        "raw_contents": raw_contents}
        

class ChatSession:
    def __init__(
            self, 
            session_id: str, 
            chat_agent_type: Literal["react", "parallel", "upload_file"]="react",
            upload_file_dir: Optional[str]=None):
        self.session_id = session_id
        self.chat_agent_type = chat_agent_type
        if session_id in chat_agents.keys():
            self.agent = chat_agents[session_id]
        else:
            self.agent = ChatAgent(
                session_id=session_id, 
                chat_agent_type=chat_agent_type,
                reader_directory=upload_file_dir)
            chat_agents[session_id] = self.agent

    @classmethod
    def get_response(
        cls, session_id: str, 
        question: str, 
        chat_agent_type: Literal["react", "parallel", "upload_file"]="react",
        upload_file_dir: Optional[str]=None):
        try:
            response = cls(session_id, chat_agent_type, upload_file_dir).agent.agent.chat(question)
            return parse_response(response)
        except Exception as e:
            return {"response": f"An exception has happened - the agent did not respond.\n{e}",
                    "sources": [],
                    "tools_used": [],
                    "raw_contents": []}
    
    def _clear_user_uploads(self, session_id: str):
        raise NotImplementedError # place holder function - not needed now
        assert self.chat_agent_type == "upload_file", "Can only clear user upload if in upload_file mode"

    @classmethod
    def clear_session(cls, session_id: str):
        chat_agents.pop(session_id)