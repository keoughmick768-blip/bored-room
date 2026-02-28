"""
The Bored Room - Core Memory System
A multi-model AI with persistent memory
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import uuid

# Storage paths
MEMORY_DIR = os.path.expanduser("~/.openclaw/memory")
os.makedirs(MEMORY_DIR, exist_ok=True)

@dataclass
class MemoryBlock:
    """A single piece of structured memory"""
    id: str
    type: str  # fact, preference, task, constraint, goal, summary
    content: str
    source: str  # which conversation
    importance: int  # 1-5
    created_at: str
    last_accessed: str
    topics: List[str]

@dataclass
class SessionState:
    """Current session state - passed to models"""
    current_goal: str = ""
    active_tasks: List[str] = None
    key_facts: List[str] = None
    constraints: List[str] = None
    decisions_made: List[str] = None
    current_focus: str = ""
    
    def __post_init__(self):
        if self.active_tasks is None:
            self.active_tasks = []
        if self.key_facts is None:
            self.key_facts = []
        if self.constraints is None:
            self.constraints = []
        if self.decisions_made is None:
            self.decisions_made = []

class MemorySystem:
    """Central memory that all models share"""
    
    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.memory_file = os.path.join(MEMORY_DIR, f"{user_id}_memory.json")
        self.session_file = os.path.join(MEMORY_DIR, f"{user_id}_session.json")
        self.conversations_file = os.path.join(MEMORY_DIR, f"{user_id}_conversations.json")
        
        self.memories: List[MemoryBlock] = []
        self.session = SessionState()
        self.conversations: Dict[str, List[Dict]] = {}
        
        self.load()
    
    def load(self):
        """Load memory from disk"""
        # Load memories
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r') as f:
                data = json.load(f)
                self.memories = [MemoryBlock(**m) for m in data]
        
        # Load session state
        if os.path.exists(self.session_file):
            with open(self.session_file, 'r') as f:
                self.session = SessionState(**json.load(f))
        
        # Load conversations
        if os.path.exists(self.conversations_file):
            with open(self.conversations_file, 'r') as f:
                self.conversations = json.load(f)
    
    def save(self):
        """Save memory to disk"""
        # Save memories
        with open(self.memory_file, 'w') as f:
            json.dump([asdict(m) for m in self.memories], f, indent=2)
        
        # Save session
        with open(self.session_file, 'w') as f:
            json.dump(asdict(self.session), f, indent=2)
        
        # Save conversations
        with open(self.conversations_file, 'w') as f:
            json.dump(self.conversations, f, indent=2)
    
    def add_memory(self, memory_type: str, content: str, source: str = "current", importance: int = 3, topics: List[str] = None):
        """Add a new memory"""
        # Check for duplicates
        for existing in self.memories:
            if existing.content.lower() == content.lower():
                existing.last_accessed = datetime.now().isoformat()
                existing.access_count = existing.__dict__.get('access_count', 0) + 1
                self.save()
                return existing
        
        memory = MemoryBlock(
            id=str(uuid.uuid4())[:8],
            type=memory_type,
            content=content,
            source=source,
            importance=importance,
            created_at=datetime.now().isoformat(),
            last_accessed=datetime.now().isoformat(),
            topics=topics or []
        )
        self.memories.append(memory)
        self.save()
        return memory
    
    def extract_and_store(self, user_message: str, assistant_message: str, conversation_id: str = "default"):
        """Extract memories from conversation"""
        # Add to conversation
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        
        self.conversations[conversation_id].append({
            "timestamp": datetime.now().isoformat(),
            "user": user_message,
            "assistant": assistant_message
        })
        
        # Extract potential memories (simple keyword-based for now)
        # In production, use an LLM to do extraction
        
        # Extract preferences (words like "prefer", "like", "hate")
        pref_keywords = ["prefer", "like", "hate", "don't like", "always", "never", "use", "working with"]
        for keyword in pref_keywords:
            if keyword in user_message.lower():
                self.add_memory("preference", user_message, source=conversation_id, importance=3)
                break
        
        # Extract tasks (words like "remind me", "don't forget", "need to")
        task_keywords = ["remind me", "don't forget", "need to", "should", "must", "have to"]
        for keyword in task_keywords:
            if keyword in user_message.lower():
                self.add_memory("task", user_message, source=conversation_id, importance=4)
                break
        
        # Extract facts (clearly stated facts)
        fact_indicators = ["i am", "i'm", "i work", "i live", "my name", "i study"]
        for indicator in fact_indicators:
            if indicator in user_message.lower():
                self.add_memory("fact", user_message, source=conversation_id, importance=4)
                break
        
        self.save()
    
    def get_relevant_memories(self, query: str, max_memories: int = 10) -> List[MemoryBlock]:
        """Get memories relevant to a query (simple keyword matching)"""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        scored_memories = []
        for memory in self.memories:
            score = 0
            # Topic match
            for topic in memory.topics:
                if topic.lower() in query_lower:
                    score += 3
            # Content match
            memory_words = set(memory.content.lower().split())
            score += len(query_words & memory_words)
            # Importance boost
            score += memory.importance
            # Recency
            if hasattr(memory, 'access_count'):
                score += min(memory.access_count, 5)
            
            if score > 0:
                scored_memories.append((score, memory))
        
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored_memories[:max_memories]]
    
    def get_structured_context(self) -> str:
        """Get memory as model-agnostic structured text"""
        context_parts = []
        
        # Session state
        if self.session.current_goal:
            context_parts.append(f"Current Goal: {self.session.current_goal}")
        if self.session.active_tasks:
            context_parts.append(f"Active Tasks: {', '.join(self.session.active_tasks)}")
        if self.session.current_focus:
            context_parts.append(f"Current Focus: {self.session.current_focus}")
        
        # Key facts
        facts = [m.content for m in self.memories if m.type == "fact"][:5]
        if facts:
            context_parts.append(f"Key Facts: {'; '.join(facts)}")
        
        # Preferences
        prefs = [m.content for m in self.memories if m.type == "preference"][:5]
        if prefs:
            context_parts.append(f"Preferences: {'; '.join(prefs)}")
        
        return "\n".join(context_parts) if context_parts else "No memory yet."
    
    def update_session(self, **kwargs):
        """Update session state"""
        for key, value in kwargs.items():
            if hasattr(self.session, key):
                setattr(self.session, key, value)
        self.save()
    
    def get_all_memories(self) -> List[MemoryBlock]:
        """Get all memories"""
        return sorted(self.memories, key=lambda x: x.created_at, reverse=True)
    
    def clear_session(self):
        """Clear session but keep long-term memory"""
        self.session = SessionState()
        self.save()


class ModelRouter:
    """Routes requests to the best model with detailed routing info"""
    
    def __init__(self):
        self.default_provider = "openclaw"
        self.default_model = "minimax/MiniMax-M2.5-Lightning"
        
        # Model database with capabilities
        self.models = {
            # OpenClaw (free local)
            "minimax/MiniMax-M2.5-Lightning": {
                "provider": "openclaw", 
                "type": "fast", 
                "strengths": ["general", "fast", "coding", "conversation"],
                "cost": "free"
            },
            # OpenAI
            "gpt-4o": {
                "provider": "openai", 
                "type": "balanced", 
                "strengths": ["reasoning", "coding", "writing", "analysis"],
                "cost": "medium"
            },
            "gpt-4o-mini": {
                "provider": "openai", 
                "type": "fast", 
                "strengths": ["fast", "simple", "conversation"],
                "cost": "low"
            },
            "o1": {
                "provider": "openai", 
                "type": "reasoning", 
                "strengths": ["reasoning", "math", "complex_analysis"],
                "cost": "high"
            },
            # Anthropic
            "claude-3-5-sonnet-20241022": {
                "provider": "anthropic", 
                "type": "balanced", 
                "strengths": ["reasoning", "coding", "writing", "analysis"],
                "cost": "medium"
            },
            "claude-3-opus-20240229": {
                "provider": "anthropic", 
                "type": "reasoning", 
                "strengths": ["reasoning", "writing", "analysis", "long_context"],
                "cost": "high"
            },
            "claude-3-sonnet-20240229": {
                "provider": "anthropic", 
                "type": "balanced", 
                "strengths": ["balanced", "coding", "writing"],
                "cost": "medium"
            },
            "claude-3-haiku-20240307": {
                "provider": "anthropic", 
                "type": "fast", 
                "strengths": ["fast", "simple", "conversation"],
                "cost": "low"
            },
            # Google
            "gemini-1.5-pro": {
                "provider": "google", 
                "type": "reasoning", 
                "strengths": ["reasoning", "analysis", "long_context"],
                "cost": "medium"
            },
            "gemini-1.5-flash": {
                "provider": "google", 
                "type": "fast", 
                "strengths": ["fast", "simple", "conversation"],
                "cost": "low"
            },
            # Mistral
            "mistral-large-latest": {
                "provider": "mistral", 
                "type": "reasoning", 
                "strengths": ["reasoning", "coding", "analysis"],
                "cost": "medium"
            },
            "mistral-small-latest": {
                "provider": "mistral", 
                "type": "fast", 
                "strengths": ["fast", "simple"],
                "cost": "low"
            },
            # Groq (fast & free tier)
            "llama-3.3-70b-versatile": {
                "provider": "groq", 
                "type": "balanced", 
                "strengths": ["balanced", "coding", "conversation"],
                "cost": "free"
            },
            # Together AI
            "meta-llama/Llama-3.3-70B-Instruct-Turbo": {
                "provider": "together", 
                "type": "reasoning", 
                "strengths": ["reasoning", "coding", "analysis"],
                "cost": "medium"
            },
            # Cohere
            "command-r-plus": {
                "provider": "cohere", 
                "type": "reasoning", 
                "strengths": ["reasoning", "analysis"],
                "cost": "medium"
            },
            # Perplexity
            "llama-3.1-sonar-small-128k-online": {
                "provider": "perplexity", 
                "type": "fast", 
                "strengths": ["research", "online", "fast"],
                "cost": "low"
            },
        }
        
        # Task patterns for routing
        self.task_patterns = {
            "coding": {
                "keywords": ["code", "program", "debug", "script", "function", "class", "python", "javascript", "html", "css", "api", "database", "build", "create file", "write code", "fix", "implement"],
                "preferred": "minimax/MiniMax-M2.5-Lightning",
                "fallback": "gpt-4o"
            },
            "reasoning": {
                "keywords": ["analyze", "explain", "compare", "research", "why", "how", "think", "reason", "logic", "solve", "math", "calculate", "complex"],
                "preferred": "claude-3-5-sonnet-20241022",
                "fallback": "gemini-1.5-pro"
            },
            "writing": {
                "keywords": ["write", "essay", "article", "blog", "post", "story", "summarize", "edit", "proofread", "draft", "creative"],
                "preferred": "claude-3-5-sonnet-20241022",
                "fallback": "gpt-4o"
            },
            "fast": {
                "keywords": ["quick", "simple", "what is", "who is", "when", "where", "hi", "hello", "hey", "thanks"],
                "preferred": "minimax/MiniMax-M2.5-Lightning",
                "fallback": "gemini-1.5-flash"
            },
            "creative": {
                "keywords": ["idea", "brainstorm", "creative", "innovate", "think of", "suggestion", "recommend", "imagine"],
                "preferred": "gpt-4o",
                "fallback": "claude-3-5-sonnet-20241022"
            },
            "memory": {
                "keywords": ["remember", "remind", "what did i say", "what do you know", "my name", "i told you", "recall"],
                "preferred": "minimax/MiniMax-M2.5-Lightning",
                "fallback": "gpt-4o-mini"
            },
            "research": {
                "keywords": ["search", "find", "look up", "latest", "news", "current", "recent", "web"],
                "preferred": "llama-3.1-sonar-small-128k-online",
                "fallback": "perplexity"
            }
        }
    
    def select_model(self, task: str, prefer_free: bool = True) -> dict:
        """Select best model for task - returns dict with provider, model, and reason"""
        task_lower = task.lower()
        
        # Score each task pattern
        pattern_scores = {}
        for pattern_name, pattern_info in self.task_patterns.items():
            score = 0
            for keyword in pattern_info["keywords"]:
                if keyword in task_lower:
                    score += 1
                    # Boost score for longer matches (more specific)
                    score += len(keyword) / 10
            if score > 0:
                pattern_scores[pattern_name] = score
        
        # Find best matching pattern
        if pattern_scores:
            best_pattern = max(pattern_scores, key=pattern_scores.get)
            pattern_info = self.task_patterns[best_pattern]
            
            # Use free model if prefer_free, otherwise use preferred
            if prefer_free:
                # Try to find a free model with the right strengths
                free_models = [
                    ("minimax/MiniMax-M2.5-Lightning", "openclaw"),
                    ("llama-3.3-70b-versatile", "groq")
                ]
                for model_id, provider in free_models:
                    return {
                        "provider": provider,
                        "model": model_id,
                        "name": model_id,
                        "reason": f"Auto-selected for {best_pattern} (free)"
                    }
                # Fallback to default free model
                return {
                    "provider": "openclaw",
                    "model": self.default_model,
                    "name": self.default_model,
                    "reason": f"Auto-selected for {best_pattern}"
                }
            else:
                model_id = pattern_info.get("preferred", self.default_model)
                model_info = self.models.get(model_id, {})
                return {
                    "provider": model_info.get("provider", "openclaw"),
                    "model": model_id,
                    "name": model_id,
                    "reason": f"Auto-selected for {best_pattern}"
                }
        
        # Default: use free model for general tasks
        return {
            "provider": "openclaw",
            "model": self.default_model,
            "name": self.default_model,
            "reason": "Default selection (general task)"
        }


# Singleton instances
_memory_system: Optional[MemorySystem] = None
_model_router: Optional[ModelRouter] = None

def get_memory_system(user_id: str = "default") -> MemorySystem:
    global _memory_system
    if _memory_system is None:
        _memory_system = MemorySystem(user_id)
    return _memory_system

def get_model_router() -> ModelRouter:
    global _model_router
    if _model_router is None:
        _model_router = ModelRouter()
    return _model_router
