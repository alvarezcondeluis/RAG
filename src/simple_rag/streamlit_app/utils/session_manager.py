"""
Session management utilities for conversation state.
"""

import json
from typing import List, Dict, Any
from datetime import datetime


class SessionManager:
    """Manage conversation session state and history."""
    
    @staticmethod
    def save_message(role: str, content: str, sources: List[Dict[str, Any]] = None):
        """
        Save a message to session history.
        
        Args:
            role: Message role (user/assistant)
            content: Message content
            sources: Optional list of source citations
        """
        import streamlit as st
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        if sources:
            message["sources"] = sources
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        st.session_state.messages.append(message)
    
    @staticmethod
    def get_conversation_context(max_messages: int = 10) -> str:
        """
        Get recent conversation context for LLM.
        
        Args:
            max_messages: Maximum number of recent messages to include
            
        Returns:
            Formatted conversation context string
        """
        import streamlit as st
        
        messages = st.session_state.get("messages", [])
        recent_messages = messages[-max_messages:] if len(messages) > max_messages else messages
        
        context_parts = []
        for msg in recent_messages:
            role = msg["role"].upper()
            content = msg["content"]
            context_parts.append(f"{role}: {content}")
        
        return "\n\n".join(context_parts)
    
    @staticmethod
    def export_to_json() -> str:
        """
        Export conversation to JSON format.
        
        Returns:
            JSON string of conversation
        """
        import streamlit as st
        
        export_data = {
            "exported_at": datetime.now().isoformat(),
            "query_count": st.session_state.get("query_count", 0),
            "messages": st.session_state.get("messages", [])
        }
        
        return json.dumps(export_data, indent=2)
    
    @staticmethod
    def clear_history():
        """Clear conversation history."""
        import streamlit as st
        
        st.session_state.messages = []
        st.session_state.query_count = 0
    
    @staticmethod
    def get_stats() -> Dict[str, Any]:
        """
        Get session statistics.
        
        Returns:
            Dictionary of session stats
        """
        import streamlit as st
        
        messages = st.session_state.get("messages", [])
        
        return {
            "total_messages": len(messages),
            "user_messages": len([m for m in messages if m["role"] == "user"]),
            "assistant_messages": len([m for m in messages if m["role"] == "assistant"]),
            "query_count": st.session_state.get("query_count", 0)
        }
