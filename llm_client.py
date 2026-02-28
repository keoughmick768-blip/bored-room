"""
LLM Integration Layer
Connects to OpenClaw, OpenAI, Claude, Google, Mistral, and many more providers
"""

import os
import json
import requests
from typing import Optional, Dict, Any, List
from datetime import datetime

# Configuration - supports both env vars and Streamlit secrets
def get_api_key(key_name: str, default: str = "") -> str:
    """Get API key from environment or Streamlit secrets"""
    # Try environment first
    env_key = os.environ.get(key_name, "")
    if env_key:
        return env_key
    
    # Try Streamlit secrets
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and key_name in st.secrets:
            return st.secrets[key_name]
    except:
        pass
    
    return default

OPENCLAW_URL = os.environ.get("OPENCLAW_URL", "http://localhost:18789")

# API Keys for all providers
OPENAI_API_KEY = get_api_key("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = get_api_key("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY = get_api_key("GOOGLE_API_KEY", "")
MISTRAL_API_KEY = get_api_key("MISTRAL_API_KEY", "")
COHERE_API_KEY = get_api_key("COHERE_API_KEY", "")
TOGETHER_API_KEY = get_api_key("TOGETHER_API_KEY", "")
GROQ_API_KEY = get_api_key("GROQ_API_KEY", "")
PERPLEXITY_API_KEY = get_api_key("PERPLEXITY_API_KEY", "")
ANTHROPIC_API_KEY = get_api_key("ANTHROPIC_API_KEY", "")
OPENROUTER_API_KEY = get_api_key("OPENROUTER_API_KEY", "")

class LLMClient:
    """Unified client for all LLM providers"""
    
    def __init__(self, provider: str = "openclaw", model: str = None):
        self.provider = provider
        self.model = model or self._default_model(provider)
    
    def _default_model(self, provider: str) -> str:
        defaults = {
            "openclaw": "minimax/MiniMax-M2.5-Lightning",
            "openai": "gpt-4o-mini",
            "anthropic": "claude-3-haiku-20240307",
            "google": "gemini-1.5-flash",
            "mistral": "mistral-small-latest",
            "cohere": "command-r-plus",
            "together": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "groq": "llama-3.3-70b-versatile",
            "perplexity": "llama-3.1-sonar-small-128k-online",
            "openrouter": "anthropic/claude-3-haiku",
        }
        return defaults.get(provider, "gpt-4o-mini")
    
    def chat(self, message: str, context: str = "", system_prompt: str = "") -> str:
        """Send a chat message and get response"""
        
        # Build full prompt with memory context
        full_prompt = self._build_prompt(message, context, system_prompt)
        
        # Route to appropriate provider
        if self.provider == "openclaw":
            return self._chat_openclaw(full_prompt)
        elif self.provider == "openai":
            return self._chat_openai(full_prompt)
        elif self.provider == "anthropic":
            return self._chat_anthropic(full_prompt)
        elif self.provider == "google":
            return self._chat_google(full_prompt)
        elif self.provider == "mistral":
            return self._chat_mistral(full_prompt)
        elif self.provider == "cohere":
            return self._chat_cohere(full_prompt)
        elif self.provider == "together":
            return self._chat_together(full_prompt)
        elif self.provider == "groq":
            return self._chat_groq(full_prompt)
        elif self.provider == "perplexity":
            return self._chat_perplexity(full_prompt)
        elif self.provider == "openrouter":
            return self._chat_openrouter(full_prompt)
        else:
            return self._chat_openclaw(full_prompt)
    
    def _build_prompt(self, message: str, context: str, system_prompt: str) -> str:
        """Build prompt with memory context"""
        parts = []
        
        if system_prompt:
            parts.append(system_prompt)
        
        if context:
            parts.append(f"\n[CONTEXT FROM MEMORY]\n{context}\n[/CONTEXT]\n")
        
        parts.append(f"\n[USER MESSAGE]\n{message}\n[/USER MESSAGE]")
        
        return "\n".join(parts)
    
    def _chat_openclaw(self, prompt: str) -> str:
        """Chat using OpenClaw local model"""
        try:
            response = requests.post(
                f"{OPENCLAW_URL}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2000
                },
                timeout=60
            )
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            pass
        
        return "OpenClaw not available. Please configure API keys for cloud models."
    
    def _chat_openai(self, prompt: str) -> str:
        """Chat using OpenAI API"""
        if not OPENAI_API_KEY:
            return "OpenAI API key not configured. Set OPENAI_API_KEY environment variable."
        
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2000
                },
                timeout=60
            )
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"OpenAI error: {response.text}"
        except Exception as e:
            return f"OpenAI error: {str(e)}"
    
    def _chat_anthropic(self, prompt: str) -> str:
        """Chat using Anthropic Claude API"""
        if not ANTHROPIC_API_KEY:
            return "Anthropic API key not configured. Set ANTHROPIC_API_KEY environment variable."
        
        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "max_tokens": 2000,
                    "messages": [{"role": "user", "content": prompt}]
                },
                timeout=60
            )
            if response.status_code == 200:
                return response.json()["content"][0]["text"]
            else:
                return f"Anthropic error: {response.text}"
        except Exception as e:
            return f"Anthropic error: {str(e)}"
    
    def _chat_google(self, prompt: str) -> str:
        """Chat using Google Gemini API"""
        if not GOOGLE_API_KEY:
            return "Google API key not configured. Set GOOGLE_API_KEY environment variable."
        
        try:
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={GOOGLE_API_KEY}",
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.7,
                        "maxOutputTokens": 2000,
                    }
                },
                timeout=60
            )
            if response.status_code == 200:
                data = response.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
            else:
                return f"Google error: {response.text}"
        except Exception as e:
            return f"Google error: {str(e)}"
    
    def _chat_mistral(self, prompt: str) -> str:
        """Chat using Mistral API"""
        if not MISTRAL_API_KEY:
            return "Mistral API key not configured. Set MISTRAL_API_KEY environment variable."
        
        try:
            response = requests.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {MISTRAL_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2000
                },
                timeout=60
            )
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"Mistral error: {response.text}"
        except Exception as e:
            return f"Mistral error: {str(e)}"
    
    def _chat_cohere(self, prompt: str) -> str:
        """Chat using Cohere API"""
        if not COHERE_API_KEY:
            return "Cohere API key not configured. Set COHERE_API_KEY environment variable."
        
        try:
            response = requests.post(
                "https://api.cohere.ai/v1/chat",
                headers={
                    "Authorization": f"Bearer {COHERE_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "message": prompt,
                    "max_tokens": 2000
                },
                timeout=60
            )
            if response.status_code == 200:
                return response.json()["text"]
            else:
                return f"Cohere error: {response.text}"
        except Exception as e:
            return f"Cohere error: {str(e)}"
    
    def _chat_together(self, prompt: str) -> str:
        """Chat using Together AI API"""
        if not TOGETHER_API_KEY:
            return "Together AI API key not configured. Set TOGETHER_API_KEY environment variable."
        
        try:
            response = requests.post(
                "https://api.together.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {TOGETHER_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2000
                },
                timeout=60
            )
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"Together AI error: {response.text}"
        except Exception as e:
            return f"Together AI error: {str(e)}"
    
    def _chat_groq(self, prompt: str) -> str:
        """Chat using Groq API"""
        if not GROQ_API_KEY:
            return "Groq API key not configured. Set GROQ_API_KEY environment variable."
        
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2000
                },
                timeout=60
            )
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"Groq error: {response.text}"
        except Exception as e:
            return f"Groq error: {str(e)}"
    
    def _chat_perplexity(self, prompt: str) -> str:
        """Chat using Perplexity API"""
        if not PERPLEXITY_API_KEY:
            return "Perplexity API key not configured. Set PERPLEXITY_API_KEY environment variable."
        
        try:
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers={
                    "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2000
                },
                timeout=60
            )
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"Perplexity error: {response.text}"
        except Exception as e:
            return f"Perplexity error: {str(e)}"
    
    def _chat_openrouter(self, prompt: str) -> str:
        """Chat using OpenRouter (aggregates many providers)"""
        if not OPENROUTER_API_KEY:
            return "OpenRouter API key not configured. Set OPENROUTER_API_KEY environment variable."
        
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://localhost:8502",
                    "X-Title": "The Bored Room"
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2000
                },
                timeout=60
            )
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"OpenRouter error: {response.text}"
        except Exception as e:
            return f"OpenRouter error: {str(e)}"
    
    @staticmethod
    def list_available_models() -> Dict[str, List[Dict]]:
        """List available models by provider"""
        return {
            "openclaw": [
                {"id": "minimax/MiniMax-M2.5-Lightning", "name": "MiniMax Lightning", "type": "fast"},
            ],
            "openai": [
                {"id": "gpt-4o", "name": "GPT-4o", "type": "balanced"},
                {"id": "gpt-4o-mini", "name": "GPT-4o Mini", "type": "fast"},
                {"id": "gpt-4-turbo", "name": "GPT-4 Turbo", "type": "balanced"},
                {"id": "o1", "name": "o1 (Reasoning)", "type": "reasoning"},
                {"id": "o1-preview", "name": "o1 Preview", "type": "reasoning"},
            ],
            "anthropic": [
                {"id": "claude-3-5-sonnet-20241022", "name": "Claude 3.5 Sonnet", "type": "balanced"},
                {"id": "claude-3-opus-20240229", "name": "Claude 3 Opus", "type": "reasoning"},
                {"id": "claude-3-sonnet-20240229", "name": "Claude 3 Sonnet", "type": "balanced"},
                {"id": "claude-3-haiku-20240307", "name": "Claude 3 Haiku", "type": "fast"},
            ],
            "google": [
                {"id": "gemini-1.5-pro", "name": "Gemini 1.5 Pro", "type": "reasoning"},
                {"id": "gemini-1.5-flash", "name": "Gemini 1.5 Flash", "type": "fast"},
                {"id": "gemini-1.5-flash-8b", "name": "Gemini 1.5 Flash 8B", "type": "fast"},
            ],
            "mistral": [
                {"id": "mistral-large-latest", "name": "Mistral Large", "type": "reasoning"},
                {"id": "mistral-small-latest", "name": "Mistral Small", "type": "fast"},
                {"id": "mistral-medium-latest", "name": "Mistral Medium", "type": "balanced"},
            ],
            "cohere": [
                {"id": "command-r-plus", "name": "Command R+", "type": "reasoning"},
                {"id": "command-r", "name": "Command R", "type": "balanced"},
                {"id": "command", "name": "Command", "type": "fast"},
            ],
            "together": [
                {"id": "meta-llama/Llama-3.3-70B-Instruct-Turbo", "name": "Llama 3.3 70B", "type": "reasoning"},
                {"id": "meta-llama/Llama-3.1-405B-Instruct-Turbo", "name": "Llama 3.1 405B", "type": "reasoning"},
                {"id": "meta-llama/Llama-3.1-70B-Instruct-Turbo", "name": "Llama 3.1 70B", "type": "balanced"},
                {"id": "mistralai/Mixtral-8x7B-Instruct-v0.1", "name": "Mixtral 8x7B", "type": "balanced"},
            ],
            "groq": [
                {"id": "llama-3.3-70b-versatile", "name": "Llama 3.3 70B Versatile", "type": "balanced"},
                {"id": "llama-3.1-70b-short", "name": "Llama 3.1 70B Short", "type": "fast"},
                {"id": "mixtral-8x7b-32768", "name": "Mixtral 8x7B", "type": "balanced"},
                {"id": "gemma2-9b-it", "name": "Gemma 2 9B", "type": "fast"},
            ],
            "perplexity": [
                {"id": "llama-3.1-sonar-large-128k-online", "name": "Sonar Large Online", "type": "reasoning"},
                {"id": "llama-3.1-sonar-small-128k-online", "name": "Sonar Small Online", "type": "fast"},
                {"id": "llama-3.1-sonar-large-128k-chat", "name": "Sonar Large Chat", "type": "reasoning"},
            ],
            "openrouter": [
                {"id": "anthropic/claude-3.5-sonnet", "name": "Claude 3.5 Sonnet", "type": "balanced"},
                {"id": "anthropic/claude-3-opus", "name": "Claude 3 Opus", "type": "reasoning"},
                {"id": "google/gemini-pro-1.5", "name": "Gemini Pro 1.5", "type": "reasoning"},
                {"id": "meta-llama/llama-3.1-70b-instruct", "name": "Llama 3.1 70B", "type": "balanced"},
                {"id": "mistralai/mistral-large", "name": "Mistral Large", "type": "reasoning"},
            ]
        }


def create_client(provider: str = "openclaw", model: str = None) -> LLMClient:
    """Factory function to create LLM client"""
    return LLMClient(provider, model)
