"""
The Bored Room - AI That Remembers Everything
Main Application with Streamlit UI
"""

import streamlit as st
import os
import sys
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory_system import get_memory_system, get_model_router, MemorySystem
from llm_client import LLMClient, create_client

# Page config
st.set_page_config(
    page_title="The Bored Room",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    .stApp {
        background: transparent;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background: rgba(0, 212, 255, 0.1);
        border-left: 3px solid #00d4ff;
    }
    .assistant-message {
        background: rgba(255, 107, 107, 0.1);
        border-left: 3px solid #ff6b6b;
    }
    .memory-tag {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        background: rgba(0, 255, 136, 0.2);
        color: #00ff88;
        border-radius: 0.25rem;
        font-size: 0.75rem;
        margin-right: 0.5rem;
    }
    .stat-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stat-value {
        font-size: 2rem;
        font-weight: bold;
        color: #00d4ff;
    }
    .stat-label {
        color: #888;
        font-size: 0.875rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory_system" not in st.session_state:
    st.session_state.memory_system = get_memory_system("mick")
if "current_model" not in st.session_state:
    st.session_state.current_model = "auto"
if "current_provider" not in st.session_state:
    st.session_state.current_provider = "auto"
if "auto_route" not in st.session_state:
    st.session_state.auto_route = True  # Default to auto-routing enabled

def send_message(message: str, model: str = None, provider: str = None):
    """Send a message and get AI response with auto-routing"""
    # Handle auto-routing
    if st.session_state.auto_route:
        router = get_model_router()
        route = router.select_model(message, prefer_free=True)
        provider = route["provider"]
        model = route["model"]
        # Track which model was selected for display
        st.session_state.last_routed_model = route["name"]
    else:
        provider = provider or st.session_state.current_provider
        model = model or st.session_state.current_model
        st.session_state.last_routed_model = None
    
    # Get memory context
    memory = st.session_state.memory_system
    context = memory.get_structured_context()
    
    # Get relevant memories
    relevant = memory.get_relevant_memories(message)
    memory_context = "\n".join([f"- {m.content} ({m.type})" for m in relevant])
    
    full_context = context
    if memory_context:
        full_context += "\n\nRelevant Memories:\n" + memory_context
    
    # Create client and send
    client = create_client(provider, model)
    
    system_prompt = """You are a helpful AI assistant with long-term memory. 
You remember facts, preferences, and context from previous conversations.
Use the provided context to give personalized responses.
Be concise but thorough."""
    
    try:
        response = client.chat(message, context=full_context, system_prompt=system_prompt)
        
        # Extract and store memories
        memory.extract_and_store(message, response)
        
        return response
    except Exception as e:
        return f"Error: {str(e)}"

def render_sidebar():
    """Render the sidebar"""
    memory = st.session_state.memory_system
    router = get_model_router()
    
    with st.sidebar:
        st.title("🧠 The Bored Room")
        st.caption("AI that remembers everything")
        
        st.divider()
        
        # Model Selection
        st.subheader("🤖 Model")
        
        # Auto-routing toggle
        auto_route = st.toggle(
            "🎯 Auto-Select Best Model",
            value=st.session_state.auto_route,
            help="Automatically selects the best model based on your task"
        )
        st.session_state.auto_route = auto_route
        
        if auto_route:
            st.success(f"🤖 Auto-routing enabled - I'll pick the best model for each task!")
            # Show what model would be selected for current message
            if st.session_state.get("last_routed_model"):
                st.info(f"Last used: {st.session_state.last_routed_model}")
        else:
            # Manual provider selection
            provider = st.selectbox(
                "Provider",
                ["openclaw", "openai", "anthropic", "google", "mistral", "groq", "together", "cohere", "perplexity", "openrouter"],
                format_func=lambda x: {
                    "openclaw": "🖥️ OpenClaw (Free)",
                    "openai": "🔷 OpenAI",
                    "anthropic": "🟣 Anthropic",
                    "google": "🔶 Google Gemini",
                    "mistral": "💜 Mistral",
                    "groq": "⚡ Groq (Fast/Free)",
                    "together": "🚀 Together AI",
                    "cohere": "🌊 Cohere",
                    "perplexity": "🔍 Perplexity",
                    "openrouter": "🌐 OpenRouter (100+ models)"
                }.get(x, x),
                index=0
            )
            st.session_state.current_provider = provider
            
            # Model selection based on provider
            models = LLMClient.list_available_models().get(provider, [])
            if models:
                model = st.selectbox(
                    "Model",
                    [m["id"] for m in models],
                    format_func=lambda x: next((m["name"] for m in models if m["id"] == x), x),
                    index=0
                )
                st.session_state.current_model = model
        
        # API Keys section
        st.subheader("🔑 API Keys")
        
        # OpenAI
        api_key_openai = st.text_input("OpenAI (sk-...)", type="password", 
            help="For GPT-4, GPT-4o models")
        if api_key_openai:
            os.environ["OPENAI_API_KEY"] = api_key_openai
        
        # Anthropic
        api_key_anthropic = st.text_input("Anthropic (sk-ant-...)", type="password",
            help="For Claude models")
        if api_key_anthropic:
            os.environ["ANTHROPIC_API_KEY"] = api_key_anthropic
        
        # Google
        api_key_google = st.text_input("Google AI", type="password",
            help="For Gemini models")
        if api_key_google:
            os.environ["GOOGLE_API_KEY"] = api_key_google
        
        # Mistral
        api_key_mistral = st.text_input("Mistral", type="password",
            help="For Mistral models")
        if api_key_mistral:
            os.environ["MISTRAL_API_KEY"] = api_key_mistral
        
        # Groq
        api_key_groq = st.text_input("Groq", type="password",
            help="Fast inference - free tier available")
        if api_key_groq:
            os.environ["GROQ_API_KEY"] = api_key_groq
        
        # Together AI
        api_key_together = st.text_input("Together AI", type="password",
            help="For Llama, Mixtral models")
        if api_key_together:
            os.environ["TOGETHER_API_KEY"] = api_key_together
        
        # Cohere
        api_key_cohere = st.text_input("Cohere", type="password",
            help="For Command R models")
        if api_key_cohere:
            os.environ["COHERE_API_KEY"] = api_key_cohere
        
        # Perplexity
        api_key_perplexity = st.text_input("Perplexity", type="password",
            help="Online research models")
        if api_key_perplexity:
            os.environ["PERPLEXITY_API_KEY"] = api_key_perplexity
        
        # OpenRouter
        api_key_openrouter = st.text_input("OpenRouter", type="password",
            help="Aggregates 100+ models")
        if api_key_openrouter:
            os.environ["OPENROUTER_API_KEY"] = api_key_openrouter
        
        st.caption("💡 Tip: Groq and OpenClaw are free!")
        
        st.divider()
        
        # Stats
        st.subheader("📊 Memory Stats")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{len(memory.memories)}</div>
                <div class="stat-label">Memories</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{len(memory.conversations.get('default', []))}</div>
                <div class="stat-label">Messages</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Quick Actions
        st.subheader("⚡ Quick Actions")
        
        if st.button("🗑️ Clear Session", use_container_width=True):
            memory.clear_session()
            st.session_state.messages = []
            st.rerun()
        
        if st.button("📋 View All Memories", use_container_width=True):
            st.session_state.show_memories = True
            st.rerun()
        
        if st.button("📤 Export Memory", use_container_width=True):
            # Export memory
            import json
            memory_data = {
                "memories": [vars(m) for m in memory.memories],
                "session": vars(memory.session),
                "exported_at": datetime.now().isoformat()
            }
            st.download_button(
                "Download JSON",
                json.dumps(memory_data, indent=2),
                "bored_room_memory.json",
                "application/json"
            )
        
        st.divider()
        
        # Memory toggle
        st.subheader("🧠 Memory")
        memory_enabled = st.toggle("Enable Memory Injection", value=True)
        if memory_enabled:
            st.caption("Memory context will be included in every response")


def render_chat():
    """Render the main chat area"""
    st.subheader("💬 Chat")
    
    # Display messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Message The Bored Room..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            # Show which model is being used
            if st.session_state.auto_route:
                if st.session_state.get("last_routed_model"):
                    st.caption(f"🎯 Auto-selected: {st.session_state.last_routed_model}")
            else:
                st.caption(f"📌 Manual: {st.session_state.current_model}")
            
            with st.spinner("Thinking..."):
                response = send_message(prompt)
                st.markdown(response)
        
        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})


def render_memories():
    """Render the memories view"""
    memory = st.session_state.memory_system
    
    st.subheader("🧠 All Memories")
    
    # Filter by type
    filter_type = st.selectbox(
        "Filter by type",
        ["all", "fact", "preference", "task", "constraint", "goal", "summary"]
    )
    
    memories = memory.get_all_memories()
    if filter_type != "all":
        memories = [m for m in memories if m.type == filter_type]
    
    if not memories:
        st.info("No memories yet. Start chatting to build memory!")
        return
    
    for m in memories:
        with st.expander(f"{m.type.upper()}: {m.content[:100]}..."):
            st.markdown(f"**Type:** {m.type}")
            st.markdown(f"**Content:** {m.content}")
            st.markdown(f"**Importance:** {'⭐' * m.importance}")
            st.markdown(f"**Created:** {m.created_at}")
            st.markdown(f"**Topics:** {', '.join(m.topics) if m.topics else 'None'}")
    
    # Clear memories
    if st.button("Clear All Memories", type="primary"):
        memory.memories = []
        memory.save()
        st.rerun()


def render_session_state():
    """Render current session state"""
    memory = st.session_state.memory_system
    session = memory.session
    
    st.subheader("📍 Current Session State")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Current Goal:**")
        new_goal = st.text_input("Goal", value=session.current_goal, key="goal_input")
        if new_goal != session.current_goal:
            memory.update_session(current_goal=new_goal)
    
    with col2:
        st.markdown("**Current Focus:**")
        new_focus = st.text_input("Focus", value=session.current_focus, key="focus_input")
        if new_focus != session.current_focus:
            memory.update_session(current_focus=new_focus)
    
    st.markdown("**Active Tasks:**")
    tasks_text = st.text_area("Tasks (one per line)", value="\n".join(session.active_tasks), height=100)
    new_tasks = [t.strip() for t in tasks_text.split("\n") if t.strip()]
    if new_tasks != session.active_tasks:
        memory.update_session(active_tasks=new_tasks)
    
    st.markdown("**Constraints:**")
    constraints_text = st.text_area("Constraints (one per line)", value="\n".join(session.constraints), height=100)
    new_constraints = [c.strip() for c in constraints_text.split("\n") if c.strip()]
    if new_constraints != session.constraints:
        memory.update_session(constraints=new_constraints)


def main():
    """Main app"""
    # Check if showing memories
    if st.session_state.get("show_memories", False):
        st.button("← Back to Chat", on_click=lambda: st.session_state.update({"show_memories": False}))
        render_memories()
        return
    
    # Check if showing session state
    if st.session_state.get("show_session", False):
        st.button("← Back to Chat", on_click=lambda: st.session_state.update({"show_session": False}))
        render_session_state()
        return
    
    # Show tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "🧠 Memories", "📍 Session"])
    
    with tab1:
        render_chat()
    
    with tab2:
        render_memories()
    
    with tab3:
        render_session_state()
    
    # Always show sidebar
    render_sidebar()


if __name__ == "__main__":
    main()
