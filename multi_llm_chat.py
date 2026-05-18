import streamlit as st
import hmac
import re
import time
import json
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any
from openai import OpenAI
import google.generativeai as genai
import base64
import traceback
from pathlib import Path
import pandas as pd

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Multi-LLM Chat", page_icon="🤖", layout="wide")

# DEFINE YOUR MODELS
AVAILABLE_MODELS = {
    "GPT-4o": {
        "api_id": "gpt-4o",
        "provider": "openai",
        "base_url": None,
        "api_key_name": "OPENAI_API_KEY",
        "price_input": 2.50,
        "price_output": 10.00,
        "icon": "🟢",
        "default_system_prompt": "You are GPT-4o, a helpful AI assistant made by OpenAI. You are helpful, academic, and structured.",
        "mention_triggers": ["gpt", "gpt4", "gpt4o", "openai"],
        "supports_vision": True
    },
    "Grok-3": {
        "api_id": "grok-3",
        "provider": "openai_compatible",
        "base_url": "https://api.x.ai/v1",
        "api_key_name": "XAI_API_KEY",
        "price_input": 3.00,
        "price_output": 15.00,
        "icon": "⚫",
        "default_system_prompt": "You are Grok 3, made by xAI. You are witty, direct, and enjoy intellectual discourse.",
        "mention_triggers": ["grok", "grok3", "xai"],
        "supports_vision": False
    },
    "Gemini 2.5 Flash": {
        "api_id": "gemini-2.5-flash",
        "provider": "google",
        "api_key_name": "GOOGLE_API_KEY",
        "price_input": 0.075,
        "price_output": 0.30,
        "icon": "🔵",
        "default_system_prompt": "You are Gemini 2.5 Flash, made by Google. You are fast, efficient, and detailed.",
        "mention_triggers": ["gemini", "google", "flash"],
        "supports_vision": True
    },
    "DeepSeek V3": {
        "api_id": "deepseek-chat",
        "provider": "openai_compatible",
        "base_url": "https://api.deepseek.com",
        "api_key_name": "DEEPSEEK_API_KEY",
        "price_input": 0.14,
        "price_output": 0.28,
        "icon": "🦈",
        "default_system_prompt": "You are DeepSeek V3, a highly capable AI assistant. You are analytical and precise.",
        "mention_triggers": ["deepseek", "ds", "deep"],
        "supports_vision": False
    }
}

# Prompt Templates
PROMPT_TEMPLATES = {
    "None": "",
    "Summarize": "Please summarize the following concisely:\n\n",
    "Explain Like I'm 5": "Explain this in simple terms a child could understand:\n\n",
    "Pros and Cons": "List the pros and cons of the following:\n\n",
    "Compare": "Compare and contrast the following items:\n\n",
    "Debate": "Present arguments for and against the following position:\n\n",
    "Code Review": "Review this code for bugs, improvements, and best practices:\n\n",
    "Translate to Python": "Convert the following to Python code:\n\n",
    "Step by Step": "Explain this step by step:\n\n"
}

SYSTEM_PROMPT_PRESETS = {
    "Default model prompts": None,
    "Concise Analyst": "You are a concise analyst. Answer directly, structure your response, and call out assumptions and uncertainty.",
    "Code Reviewer": "You are a senior software engineer. Review code for correctness, safety, performance, maintainability, and deployment risk. Prioritize actionable fixes.",
    "Academic Tutor": "You are an academic tutor. Explain clearly, define terms, and provide examples while avoiding unnecessary jargon.",
    "Executive Summary": "You are an executive briefing assistant. Summarize key points, risks, decisions, and recommended next actions.",
    "Adversarial Reviewer": "You are a careful adversarial reviewer. Identify weaknesses, edge cases, hidden assumptions, and counterarguments constructively."
}

HISTORY_LIMIT_OPTIONS = {
    "Last 5 messages": 5,
    "Last 10 messages": 10,
    "Last 20 messages": 20,
    "All messages": None,
}

GEMINI_SAFETY_MODES = ["Default", "Relaxed", "Block none"]

# --- 2. SESSION STATE INITIALIZATION ---
def init_session_state():
    """Initialize session state with default values."""
    defaults = {
        "messages": [],
        "session_cost": 0.0,
        "total_tokens": 0,
        "custom_prompts": {name: config["default_system_prompt"] for name, config in AVAILABLE_MODELS.items()},
        "message_id_counter": 0,
        "uploaded_image": None,
        "uploaded_image_b64": None,
        "uploaded_image_mime": None,
        "password_correct": None,
        "conversation_title": "Untitled Chat",
        "ad_hoc_model_configs": {},
        "ad_hoc_api_keys": {},
        "response_batches": [],
        "last_batch_id": 0
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

APP_VERSION = "v3-model-lab"


def ensure_v3_state() -> None:
    """Initialize v3-specific state safely for existing deployments."""
    st.session_state.setdefault("ad_hoc_model_configs", {})
    st.session_state.setdefault("ad_hoc_api_keys", {})
    st.session_state.setdefault("response_batches", [])
    st.session_state.setdefault("last_batch_id", 0)


ensure_v3_state()


def slugify_model_name(name: str) -> str:
    """Create a safe stable suffix for dynamic/ad-hoc model config keys."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", name.strip()).strip("_").upper()
    return slug or "ADHOC_MODEL"


def normalize_model_config(name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure every static or ad-hoc model has the fields the app expects."""
    cfg = dict(config)
    cfg.setdefault("display_name", name)
    cfg.setdefault("api_id", name)
    cfg.setdefault("provider", "openai_compatible")
    cfg.setdefault("base_url", None)
    cfg.setdefault("api_key_name", f"ADHOC_{slugify_model_name(name)}_API_KEY")
    cfg.setdefault("price_input", 0.0)
    cfg.setdefault("price_output", 0.0)
    cfg.setdefault("icon", "🧪")
    if "family" not in cfg:
        if cfg.get("provider") == "google":
            cfg["family"] = "Gemini"
        elif "grok" in name.lower():
            cfg["family"] = "Grok"
        elif "deepseek" in name.lower():
            cfg["family"] = "DeepSeek"
        elif "gpt" in name.lower() or cfg.get("provider") == "openai":
            cfg["family"] = "OpenAI"
        else:
            cfg["family"] = "Ad-hoc"
    cfg.setdefault("generation", "current" if not cfg.get("is_ad_hoc") else "custom")
    cfg.setdefault("default_system_prompt", f"You are {name}, a helpful AI assistant.")
    cfg.setdefault("mention_triggers", [slugify_model_name(name).lower()])
    cfg.setdefault("supports_vision", False)
    cfg.setdefault("is_ad_hoc", False)
    return cfg


def get_available_models() -> Dict[str, Dict[str, Any]]:
    """Return the active model registry, including session-added ad-hoc models."""
    registry = {name: normalize_model_config(name, cfg) for name, cfg in AVAILABLE_MODELS.items()}
    for name, cfg in st.session_state.get("ad_hoc_model_configs", {}).items():
        registry[name] = normalize_model_config(name, cfg)
    return registry


def get_model_config(model_name: str) -> Dict[str, Any]:
    """Fetch config from the live registry."""
    return get_available_models()[model_name]


def get_model_identity_label(model_name: str) -> str:
    """Human-readable identity line for UI/debug/export."""
    cfg = get_model_config(model_name)
    base = cfg.get("base_url") or "default provider endpoint"
    return f"{cfg.get('display_name', model_name)} → {cfg.get('api_id')} · {cfg.get('provider')} · {base}"


def next_batch_id() -> int:
    st.session_state.last_batch_id = int(st.session_state.get("last_batch_id", 0)) + 1
    return st.session_state.last_batch_id


def is_error_content(content: str) -> bool:
    return str(content or "").strip().startswith("⚠️")


def summarize_response_rows(batch: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Rows for the comparison table."""
    rows = []
    for item in batch.get("responses", []):
        text = item.get("content", "") or ""
        rows.append({
            "Model": item.get("name", "Unknown"),
            "Status": "Failed" if item.get("is_error") else "OK",
            "Model ID": item.get("model_id", ""),
            "Provider": item.get("provider", ""),
            "Time (s)": round(float(item.get("response_time") or 0), 2),
            "Tokens In": int(item.get("input_tokens") or 0),
            "Tokens Out": int(item.get("output_tokens") or 0),
            "Cost": round(float(item.get("estimated_cost") or 0), 6),
            "Chars": len(text),
            "Preview": text.replace("\n", " ")[:160]
        })
    return rows


def compute_model_performance(messages: List[Dict[str, Any]]) -> pd.DataFrame:
    """Aggregate model performance from assistant messages in this session."""
    records = []
    for msg in messages:
        if msg.get("role") != "assistant" or msg.get("is_judge"):
            continue
        records.append({
            "Model": msg.get("name", "Unknown"),
            "Calls": 1,
            "Failures": 1 if msg.get("is_error") else 0,
            "Total Seconds": float(msg.get("response_time") or 0),
            "Tokens": int(msg.get("input_tokens") or 0) + int(msg.get("output_tokens") or 0),
            "Cost": float(msg.get("estimated_cost") or 0),
        })
    if not records:
        return pd.DataFrame(columns=["Model", "Calls", "Failures", "Failure Rate", "Avg Seconds", "Tokens", "Cost"])
    df = pd.DataFrame(records).groupby("Model", as_index=False).sum(numeric_only=True)
    df["Failure Rate"] = (df["Failures"] / df["Calls"]).round(3)
    df["Avg Seconds"] = (df["Total Seconds"] / df["Calls"]).round(2)
    df["Cost"] = df["Cost"].round(6)
    return df[["Model", "Calls", "Failures", "Failure Rate", "Avg Seconds", "Tokens", "Cost"]]

# --- 3. AUTHENTICATION ---
def check_password() -> bool:
    """Handle password authentication with constant-time comparison."""
    def password_entered():
        entered = st.session_state.get("password", "")
        correct = st.secrets.get("APP_PASSWORD", "")
        if entered and correct and hmac.compare_digest(entered, correct):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False
            if "password" in st.session_state:
                del st.session_state["password"]

    if st.session_state.get("password_correct", False):
        return True

    st.title("🔒 Login Required")
    st.text_input(
        "Password:", 
        type="password", 
        on_change=password_entered, 
        key="password"
    )
    
    if st.session_state.get("password_correct") is False:
        st.error("😕 Password incorrect")
    return False

if not check_password():
    st.stop()

st.title("🤖 Multi-Model Arena")

# --- APP INSTRUCTIONS ---
with st.expander("📚 How to use this app", expanded=False):
    st.markdown("""
    **Getting Started:**
    - Select which AI models to chat with using the sidebar
    - All selected models respond by default
    
    **Directed Chat:**
    - Use `@gpt`, `@grok`, `@gemini`, or `@deepseek` to target specific models
    - Example: `@grok tell me a joke`
    
    **Features:**
    - 📎 Upload images for vision-capable models
    - 🔀 Side-by-side view for easy comparison
    - 💬 Debate mode: models respond to each other
    - 📥 Export your conversation as Markdown or JSON
    - 🔄 Retry any response
    - ⚡ Streaming responses for real-time output
    
    **Tips:**
    - Adjust temperature for more creative or focused responses
    - Use prompt templates for common tasks
    - Customize system prompts per model in Settings
    """)

# --- 4. HELPER FUNCTIONS ---

@st.cache_resource
def get_openai_client(api_key: str, base_url: Optional[str] = None, timeout: int = 60) -> OpenAI:
    """Cache OpenAI-compatible clients to avoid recreation."""
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

def get_api_key(key_name: str, user_provided_key: Optional[str] = None) -> Optional[str]:
    """Get API key from user input or secrets with fallback."""
    if user_provided_key:
        return user_provided_key
    return st.secrets.get(key_name)

def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate API cost based on token usage."""
    registry = get_available_models()
    if model_name not in registry:
        return 0.0
    info = registry[model_name]
    in_cost = (input_tokens / 1_000_000) * info["price_input"]
    out_cost = (output_tokens / 1_000_000) * info["price_output"]
    return in_cost + out_cost

def get_next_message_id() -> int:
    """Generate unique message ID."""
    st.session_state.message_id_counter += 1
    return st.session_state.message_id_counter

def parse_mentions(user_input: str, active_agents: List['Agent']) -> List['Agent']:
    """Parse @mentions to determine which agents should respond."""
    lower_input = user_input.lower()
    mentioned_agents = []
    
    for agent in active_agents:
        triggers = agent.config.get("mention_triggers", [])
        for trigger in triggers:
            if f"@{trigger}" in lower_input:
                if agent not in mentioned_agents:
                    mentioned_agents.append(agent)
                break
    
    return mentioned_agents if mentioned_agents else active_agents

def export_chat_markdown() -> str:
    """Export chat history as Markdown with metadata."""
    lines = [
        f"# {st.session_state.get('conversation_title', 'Chat Export')}",
        f"*Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        ""
    ]
    
    for msg in st.session_state.messages:
        name = msg.get("name", "Unknown")
        content = msg.get("content", "")
        role_icon = "👤" if msg["role"] == "user" else msg.get("avatar", "🤖")
        lines.append(f"### {role_icon} {name}")
        lines.append(content)
        if msg.get("response_time"):
            meta_parts = [f"Response time: {msg['response_time']:.2f}s"]
            if msg.get("model_id"):
                meta_parts.append(f"Model: {msg.get('model_id')}")
            if msg.get("input_tokens") is not None and msg.get("output_tokens") is not None:
                meta_parts.append(f"Tokens: {msg.get('input_tokens', 0):,} in / {msg.get('output_tokens', 0):,} out")
            if msg.get("estimated_cost") is not None:
                meta_parts.append(f"Estimated cost: ${msg.get('estimated_cost', 0.0):.5f}")
            lines.append("*" + " · ".join(meta_parts) + "*")
        if msg.get("is_debate"):
            lines.append("*[Debate Round]*")
        lines.append("")
    
    lines.extend([
        "---",
        f"**Total Cost:** ${st.session_state.session_cost:.5f}",
        f"**Total Tokens:** {st.session_state.total_tokens:,}"
    ])
    
    return "\n".join(lines)

def export_chat_json() -> str:
    """Export chat history as JSON with full metadata."""
    export_data = {
        "title": st.session_state.get("conversation_title", "Untitled Chat"),
        "exported_at": datetime.now().isoformat(),
        "session_cost": st.session_state.session_cost,
        "total_tokens": st.session_state.total_tokens,
        "app_version": APP_VERSION,
        "model_registry": {name: {k: v for k, v in cfg.items() if k != "api_key"} for name, cfg in get_available_models().items()},
        "response_batches": st.session_state.get("response_batches", []),
        "messages": st.session_state.messages
    }
    return json.dumps(export_data, indent=2)

def encode_image_to_base64(uploaded_file) -> Optional[str]:
    """Encode uploaded image to base64 string."""
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        return base64.standard_b64encode(bytes_data).decode("utf-8")
    return None

def safe_get_image_mime_type(uploaded_file) -> str:
    """Safely determine image MIME type from uploaded file."""
    if uploaded_file is None:
        return "image/jpeg"
    
    file_type = uploaded_file.type
    if file_type:
        return file_type
    
    # Fallback: detect from filename
    name = uploaded_file.name.lower()
    if name.endswith('.png'):
        return "image/png"
    elif name.endswith('.gif'):
        return "image/gif"
    elif name.endswith('.webp'):
        return "image/webp"
    else:
        return "image/jpeg"


def get_limited_history(messages: List[Dict], limit: Optional[int]) -> List[Dict]:
    """Return recent conversation messages, preserving all messages when limit is None."""
    if limit is None or limit <= 0:
        return messages
    return messages[-limit:]


def get_safe_api_error(provider_name: str = "API") -> str:
    """User-facing error text that avoids exposing raw provider internals."""
    return f"⚠️ {provider_name} request failed. Check the app logs or provider dashboard for details."


def log_exception(context: str, exc: Exception) -> None:
    """Print detailed errors to Streamlit Cloud logs without exposing them in the UI."""
    print(f"[{datetime.now().isoformat()}] {context}: {exc}")
    print(traceback.format_exc())


def build_gemini_safety_settings(mode: str) -> Optional[Dict[str, str]]:
    """Return Gemini safety settings for the selected mode."""
    if mode == "Block none":
        return {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
        }
    if mode == "Relaxed":
        return {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_ONLY_HIGH",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_ONLY_HIGH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_ONLY_HIGH",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_ONLY_HIGH",
        }
    return None


def model_has_api_key(model_name: str, user_keys: Dict[str, str]) -> bool:
    """Check whether a selected model has a key from secrets or sidebar input."""
    cfg = get_model_config(model_name)
    key_name = cfg["api_key_name"]
    if cfg.get("is_ad_hoc") and st.session_state.get("ad_hoc_api_keys", {}).get(model_name):
        return True
    return bool(user_keys.get(model_name) or st.secrets.get(key_name))


def add_assistant_message(agent: 'Agent', content: str, elapsed: float, in_tok: int, out_tok: int, *, is_debate: bool = False, is_judge: bool = False, batch_id: Optional[int] = None) -> int:
    """Append an assistant message with export-friendly metadata."""
    cost = calculate_cost(agent.name, in_tok, out_tok)
    msg_id = get_next_message_id()
    st.session_state.messages.append({
        "id": msg_id,
        "role": "assistant",
        "name": agent.name,
        "content": content,
        "avatar": agent.avatar,
        "response_time": elapsed,
        "provider": agent.provider,
        "model_id": agent.model_id,
        "display_name": agent.config.get("display_name", agent.name),
        "base_url": agent.config.get("base_url"),
        "family": agent.config.get("family", ""),
        "generation": agent.config.get("generation", ""),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "estimated_cost": cost,
        "is_debate": is_debate,
        "is_judge": is_judge,
        "is_error": is_error_content(content),
        "batch_id": batch_id,
    })
    st.session_state.session_cost += cost
    st.session_state.total_tokens += (in_tok + out_tok)
    return msg_id

# --- 5. AGENT CLASS ---
class Agent:
    """AI Agent that handles model interactions."""
    
    KNOWN_AI_NAMES = [
        "User", "Human", "Claude", "Anthropic", "Dall-E", "Bard", "Bing",
        "GPT", "GPT-4", "GPT-4o", "ChatGPT", "OpenAI",
        "Grok", "Grok-3", "xAI",
        "DeepSeek", "DeepSeek V3",
        "Gemini", "Google", "Gemini 2.5 Flash",
        "Llama", "Mistral", "Assistant", "AI", "Bot"
    ]
    
    def __init__(
        self, 
        display_name: str, 
        config: Dict[str, Any], 
        user_key: Optional[str] = None,
        concise_mode: bool = True,
        temperature: float = 0.7,
        custom_system_prompt: Optional[str] = None,
        gemini_safety_mode: str = "Default",
        request_timeout: int = 60
    ):
        self.name = display_name
        self.config = config
        self.model_id = config["api_id"]
        self.provider = config["provider"]
        self.avatar = config["icon"]
        self.temperature = max(0.0, min(2.0, temperature))  # Clamp temperature
        self.supports_vision = config.get("supports_vision", False)
        self.gemini_safety_mode = gemini_safety_mode
        self.request_timeout = max(10, min(180, int(request_timeout)))
        
        # Build system prompt
        base_prompt = custom_system_prompt or config["default_system_prompt"]
        if concise_mode:
            base_prompt += (
                "\n\nIMPORTANT INSTRUCTIONS:"
                "\n- Be brief, concise, and direct in your responses."
                "\n- Do not prefix your response with your name."
                "\n- Do not roleplay as or speak for other AI models."
                "\n- Only provide detailed explanations when explicitly requested."
            )
        self.system_prompt = base_prompt
        
        # Get API key
        self.api_key = get_api_key(config["api_key_name"], user_key)
        self.error = None if self.api_key else f"Missing API Key: {config['api_key_name']}"

    def _clean_response(self, text: str) -> str:
        """Clean model response of identity headers and artifacts."""
        if not text or not text.strip():
            return "..."
        
        text = text.strip()
        
        # Build pattern for known AI names
        escaped_names = [re.escape(name) for name in self.KNOWN_AI_NAMES]
        escaped_names.extend([re.escape(name) for name in get_available_models().keys()])
        names_pattern = "|".join(escaped_names)
        
        # Remove leading identity markers
        patterns_to_remove = [
            rf"^\[?({names_pattern})\]?\s*[:\-]\s*",
            rf"^\*\*({names_pattern})\*\*\s*[:\-]?\s*",
        ]
        for pattern in patterns_to_remove:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
        
        # Remove mid-text identity continuation
        continuation_pattern = rf"\n\s*\[?({names_pattern})\]?\s*[:\-]"
        match = re.search(continuation_pattern, text, flags=re.IGNORECASE)
        if match:
            text = text[:match.start()].strip()
        
        # Remove markdown identity continuation
        md_continuation = rf"\n\s*\*\*({names_pattern})\*\*\s*[:\-]?"
        match = re.search(md_continuation, text, flags=re.IGNORECASE)
        if match:
            text = text[:match.start()].strip()
        
        return text if text else "..."

    def _format_history_message(self, msg: Dict) -> str:
        """Format a history message with clear speaker attribution."""
        speaker = msg.get('name', 'Unknown')
        content = msg.get('content', '')
        return f"[{speaker}]: {content}"

    def generate_response_streaming(
        self, 
        conversation_history: List[Dict],
        placeholder,
        image_b64: Optional[str] = None,
        image_mime: str = "image/jpeg"
    ) -> Tuple[str, int, int, float]:
        """Generate streaming response. Returns (content, in_tokens, out_tokens, time)."""
        
        if self.error:
            return f"⚠️ {self.error}", 0, 0, 0.0

        start_time = time.time()
        
        try:
            if self.provider == "google":
                content, in_tok, out_tok = self._stream_google(
                    conversation_history, placeholder, image_b64, image_mime
                )
            else:
                content, in_tok, out_tok = self._stream_openai_compatible(
                    conversation_history, placeholder, image_b64, image_mime
                )
            
            elapsed = time.time() - start_time
            return self._clean_response(content), in_tok, out_tok, elapsed
            
        except Exception as e:
            elapsed = time.time() - start_time
            log_exception(f"Error in {self.name}", e)
            return get_safe_api_error(self.name), 0, 0, elapsed

    def _stream_openai_compatible(
        self, 
        history: List[Dict], 
        placeholder,
        image_b64: Optional[str] = None,
        image_mime: str = "image/jpeg"
    ) -> Tuple[str, int, int]:
        """Stream from OpenAI-compatible API with improved error handling."""
        
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Build conversation history
        for msg in history:
            role = "user" if msg["role"] == "user" else "assistant"
            content = self._format_history_message(msg)
            messages.append({"role": role, "content": content})

        # Add image to last user message if provided and supported
        if image_b64 and self.supports_vision and messages:
            last_user_idx = None
            for i in range(len(messages) - 1, -1, -1):
                if messages[i]["role"] == "user":
                    last_user_idx = i
                    break
            
            if last_user_idx is not None:
                text_content = messages[last_user_idx]["content"]
                messages[last_user_idx]["content"] = [
                    {"type": "text", "text": text_content},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{image_mime};base64,{image_b64}"}
                    }
                ]

        try:
            client = get_openai_client(self.api_key, self.config.get("base_url"), self.request_timeout)
            
            stream = client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=self.temperature,
                max_tokens=2048,
                stream=True
            )
            
            full_response = ""
            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta_content = chunk.choices[0].delta.content
                    if delta_content:
                        full_response += delta_content
                        placeholder.markdown(f"**{self.name}**: {full_response}▌")
            
            placeholder.markdown(f"**{self.name}**: {full_response}")
            
            # Estimate tokens (streaming doesn't always return usage)
            in_tok = sum(len(str(m.get("content", ""))) // 4 for m in messages)
            out_tok = len(full_response) // 4
            
            return full_response, in_tok, out_tok
            
        except Exception as e:
            error_msg = str(e).lower()
            if "rate" in error_msg and "limit" in error_msg:
                return "⚠️ Rate limit reached. Please wait and try again.", 0, 0
            elif "invalid" in error_msg and "key" in error_msg:
                return "⚠️ Invalid API key.", 0, 0
            elif "quota" in error_msg or "insufficient" in error_msg:
                return "⚠️ API quota exceeded. Check your billing.", 0, 0
            log_exception(f"{self.name} API error", e)
            return get_safe_api_error(self.name), 0, 0

    def _stream_google(
        self, 
        history: List[Dict], 
        placeholder,
        image_b64: Optional[str] = None,
        image_mime: str = "image/jpeg"
    ) -> Tuple[str, int, int]:
        """Stream from Google Gemini API with improved fallback handling."""
        
        try:
            genai.configure(api_key=self.api_key)
        except Exception as e:
            log_exception(f"{self.name} Google configuration error", e)
            return get_safe_api_error(self.name), 0, 0
        
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=2048,
            temperature=self.temperature
        )
        
        try:
            model = genai.GenerativeModel(
                self.model_id,
                system_instruction=self.system_prompt,
                generation_config=generation_config
            )
        except Exception as e:
            log_exception(f"{self.name} Google model initialization error", e)
            return get_safe_api_error(self.name), 0, 0
        
        # Build Google-formatted history
        google_history = []
        for msg in history:
            role = "user" if msg["role"] == "user" else "model"
            content = self._format_history_message(msg)
            google_history.append({"role": role, "parts": [content]})

        # Add image to last user message if provided
        if image_b64 and self.supports_vision and google_history:
            for i in range(len(google_history) - 1, -1, -1):
                if google_history[i]["role"] == "user":
                    google_history[i]["parts"].append({
                        "inline_data": {
                            "mime_type": image_mime,
                            "data": image_b64
                        }
                    })
                    break

        safety_settings = build_gemini_safety_settings(self.gemini_safety_mode)
        
        # Try streaming first
        try:
            response = model.generate_content(
                google_history,
                safety_settings=safety_settings,
                stream=True
            )
            
            full_response = ""
            chunk_count = 0
            
            for chunk in response:
                try:
                    chunk_text = chunk.text
                    if chunk_text:
                        full_response += chunk_text
                        placeholder.markdown(f"**{self.name}**: {full_response}▌")
                        chunk_count += 1
                except (ValueError, AttributeError) as chunk_error:
                    # Some chunks may not have text, continue
                    continue
            
            # Handle empty response
            if not full_response.strip():
                try:
                    if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                        full_response = "⚠️ Response was blocked by safety filters."
                    else:
                        full_response = "⚠️ Received your message but generated an empty response."
                except:
                    full_response = "⚠️ Received your message but generated an empty response."
            
            placeholder.markdown(f"**{self.name}**: {full_response}")
            
            # Extract token usage
            try:
                usage = response.usage_metadata
                in_tok = getattr(usage, 'prompt_token_count', len(str(google_history)) // 4)
                out_tok = getattr(usage, 'candidates_token_count', len(full_response) // 4)
            except:
                in_tok = len(str(google_history)) // 4
                out_tok = len(full_response) // 4
            
            return full_response, in_tok, out_tok
            
        except Exception as e:
            error_str = str(e).lower()
            
            # Handle rate limiting - try non-streaming fallback
            if "429" in str(e) or "quota" in error_str or "rate" in error_str:
                return self._call_google_non_streaming(
                    google_history, safety_settings, placeholder, model
                )
            
            # Handle model not found
            if "404" in error_str or "not found" in error_str:
                return self._google_fallback_different_model(
                    google_history, safety_settings, placeholder, str(e)
                )
            
            log_exception(f"{self.name} Google API error", e)
            return get_safe_api_error(self.name), 0, 0

    def _call_google_non_streaming(
        self,
        google_history: List[Dict],
        safety_settings: Dict,
        placeholder,
        model
    ) -> Tuple[str, int, int]:
        """Non-streaming fallback for Gemini when rate-limited."""
        
        fallback_models = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"]
        
        for fallback_model_name in fallback_models:
            try:
                placeholder.markdown(f"**{self.name}**: *Trying {fallback_model_name}...*")
                
                fallback_model = genai.GenerativeModel(
                    fallback_model_name,
                    system_instruction=self.system_prompt
                )
                
                response = fallback_model.generate_content(
                    google_history,
                    safety_settings=safety_settings
                )
                
                try:
                    text = response.text
                except ValueError:
                    continue
                
                if not text or not text.strip():
                    continue
                
                text += f"\n\n*(Used {fallback_model_name} due to rate limits)*"
                placeholder.markdown(f"**{self.name}**: {text}")
                
                usage = response.usage_metadata
                in_tok = getattr(usage, 'prompt_token_count', 0)
                out_tok = getattr(usage, 'candidates_token_count', 0)
                
                return text, in_tok, out_tok
                
            except Exception as fallback_e:
                fallback_error = str(fallback_e).lower()
                # If rate limited, try next model
                if "429" in str(fallback_e) or "quota" in fallback_error:
                    continue
                continue
        
        # All fallbacks failed
        return (
            "⚠️ All Gemini models are currently rate-limited. "
            "Please wait a minute and try again, or check your "
            "[quota](https://ai.google.dev/gemini-api/docs/rate-limits)."
        ), 0, 0

    def _google_fallback_different_model(
        self,
        google_history: List[Dict],
        safety_settings: Dict,
        placeholder,
        original_error: str
    ) -> Tuple[str, int, int]:
        """Try different Gemini model when primary model not found."""
        
        fallback_models = ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"]
        
        for fallback_model_name in fallback_models:
            try:
                placeholder.markdown(f"**{self.name}**: *Trying {fallback_model_name}...*")
                
                model = genai.GenerativeModel(
                    fallback_model_name,
                    system_instruction=self.system_prompt
                )
                
                response = model.generate_content(
                    google_history,
                    safety_settings=safety_settings,
                    stream=True
                )
                
                full_response = ""
                for chunk in response:
                    try:
                        chunk_text = chunk.text
                        if chunk_text:
                            full_response += chunk_text
                            placeholder.markdown(f"**{self.name}**: {full_response}▌")
                    except (ValueError, AttributeError):
                        continue
                
                if full_response.strip():
                    full_response += f"\n\n*(Used {fallback_model_name} as fallback)*"
                    placeholder.markdown(f"**{self.name}**: {full_response}")
                    
                    try:
                        usage = response.usage_metadata
                        in_tok = getattr(usage, 'prompt_token_count', 0)
                        out_tok = getattr(usage, 'candidates_token_count', 0)
                    except:
                        in_tok = len(str(google_history)) // 4
                        out_tok = len(full_response) // 4
                    
                    return full_response, in_tok, out_tok
                    
            except Exception:
                continue
        
        print(f"Gemini model fallback failed. Original error: {original_error}")
        return get_safe_api_error(self.name), 0, 0


# --- 6. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")

    st.session_state.conversation_title = st.text_input(
        "Conversation title",
        value=st.session_state.get("conversation_title", "Untitled Chat"),
        help="Used in Markdown/JSON exports."
    )

    with st.expander("🧪 Add ad-hoc OpenAI-compatible model", expanded=False):
        st.caption("Use this for OpenAI-compatible providers such as OpenRouter, Together, Fireworks, local gateways, or custom endpoints. Keys are stored only in this Streamlit session unless you add them to secrets.")
        adhoc_name = st.text_input("Display name", placeholder="My Custom Model", key="adhoc_display_name")
        adhoc_model_id = st.text_input("API model ID", placeholder="provider/model-id", key="adhoc_model_id")
        adhoc_base_url = st.text_input("Base URL", placeholder="https://api.example.com/v1", key="adhoc_base_url")
        adhoc_key_name = st.text_input("Optional Streamlit secret name", placeholder="MY_PROVIDER_API_KEY", key="adhoc_key_name")
        adhoc_api_key = st.text_input("API key for this session", type="password", key="adhoc_api_key")
        adhoc_icon = st.text_input("Icon", value="🧪", max_chars=2, key="adhoc_icon")
        adhoc_supports_vision = st.checkbox("Supports OpenAI-style vision messages", value=False, key="adhoc_vision")
        col_price1, col_price2 = st.columns(2)
        with col_price1:
            adhoc_price_in = st.number_input("$/1M input", min_value=0.0, value=0.0, step=0.01, key="adhoc_price_in")
        with col_price2:
            adhoc_price_out = st.number_input("$/1M output", min_value=0.0, value=0.0, step=0.01, key="adhoc_price_out")
        if st.button("Add/update ad-hoc model", use_container_width=True):
            if not adhoc_name.strip() or not adhoc_model_id.strip():
                st.warning("Display name and API model ID are required.")
            else:
                key_name = adhoc_key_name.strip() or f"ADHOC_{slugify_model_name(adhoc_name)}_API_KEY"
                st.session_state.ad_hoc_model_configs[adhoc_name.strip()] = {
                    "display_name": adhoc_name.strip(),
                    "api_id": adhoc_model_id.strip(),
                    "provider": "openai_compatible",
                    "base_url": adhoc_base_url.strip() or None,
                    "api_key_name": key_name,
                    "price_input": float(adhoc_price_in),
                    "price_output": float(adhoc_price_out),
                    "icon": adhoc_icon or "🧪",
                    "family": "Ad-hoc",
                    "generation": "custom",
                    "default_system_prompt": f"You are {adhoc_name.strip()}, a helpful AI assistant.",
                    "mention_triggers": [slugify_model_name(adhoc_name).lower(), adhoc_name.lower().replace(" ", "")],
                    "supports_vision": bool(adhoc_supports_vision),
                    "is_ad_hoc": True,
                }
                if adhoc_api_key:
                    st.session_state.ad_hoc_api_keys[adhoc_name.strip()] = adhoc_api_key
                st.success(f"Added {adhoc_name.strip()}.")
                st.rerun()
        if st.session_state.get("ad_hoc_model_configs"):
            st.caption("Current ad-hoc models:")
            for adhoc_existing in list(st.session_state.ad_hoc_model_configs.keys()):
                c_name, c_remove = st.columns([0.72, 0.28])
                c_name.caption(adhoc_existing)
                if c_remove.button("Remove", key=f"remove_adhoc_{adhoc_existing}"):
                    st.session_state.ad_hoc_model_configs.pop(adhoc_existing, None)
                    st.session_state.ad_hoc_api_keys.pop(adhoc_existing, None)
                    st.rerun()

    all_models = get_available_models()

    # Model selection
    selected_models = st.multiselect(
        "Select Participants:",
        options=list(all_models.keys()),
        default=[m for m in ["GPT-4o", "Grok-3"] if m in all_models],
        help="Choose which AI models will participate"
    )
    
    if not selected_models:
        st.warning("⚠️ Please select at least one model.")

    st.divider()
    
    # Settings Section
    st.subheader("Settings")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        concise_mode = st.checkbox("Concise", value=True, help="Brief responses")
    with col_s2:
        side_by_side = st.checkbox("Side-by-Side", value=False, help="Compare responses")
    
    debate_mode = st.checkbox(
        "🔄 Debate Mode", 
        value=False, 
        help="Models respond to each other after initial response"
    )

    judge_mode = st.checkbox(
        "⚖️ Judge Mode",
        value=False,
        help="Ask one model to compare the responses and produce a combined assessment"
    )

    judge_model_name = None
    if judge_mode and selected_models:
        judge_model_name = st.selectbox(
            "Judge model",
            options=selected_models,
            index=0,
            help="The selected model will evaluate the latest model responses."
        )

    history_choice = st.selectbox(
        "Conversation memory",
        options=list(HISTORY_LIMIT_OPTIONS.keys()),
        index=2,
        help="Limit how much chat history is sent to providers. This can reduce cost, latency, and context-limit errors."
    )
    history_limit = HISTORY_LIMIT_OPTIONS[history_choice]
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="Higher = more creative, Lower = more focused"
    )

    request_timeout = st.slider(
        "Provider timeout, seconds",
        min_value=10,
        max_value=180,
        value=60,
        step=10,
        help="Applied to OpenAI-compatible providers. Gemini streaming may use the Google SDK defaults."
    )

    gemini_safety_mode = st.selectbox(
        "Gemini safety mode",
        options=GEMINI_SAFETY_MODES,
        index=0,
        help="Default uses Google's default safety behavior. Relaxed blocks only high-confidence harms. Block none matches the prior app behavior."
    )
    
    # Prompt Templates
    st.divider()
    st.subheader("Prompt Template")
    selected_template = st.selectbox(
        "Quick prompts:",
        options=list(PROMPT_TEMPLATES.keys()),
        index=0,
        label_visibility="collapsed"
    )
    
    # Custom System Prompts
    with st.expander("🎛️ Custom System Prompts"):
        st.caption("Customize each model's personality")
        preset_name = st.selectbox("Preset", options=list(SYSTEM_PROMPT_PRESETS.keys()))
        if st.button("Apply preset to selected models", use_container_width=True):
            preset_prompt = SYSTEM_PROMPT_PRESETS[preset_name]
            for model_name in selected_models:
                if preset_prompt is None:
                    st.session_state.custom_prompts[model_name] = all_models[model_name]["default_system_prompt"]
                else:
                    st.session_state.custom_prompts[model_name] = preset_prompt
            st.rerun()

        for model_name in selected_models:
            new_prompt = st.text_area(
                f"{all_models[model_name]['icon']} {model_name}",
                value=st.session_state.custom_prompts.get(
                    model_name, 
                    all_models[model_name]["default_system_prompt"]
                ),
                height=80,
                key=f"prompt_{model_name}"
            )
            st.session_state.custom_prompts[model_name] = new_prompt
    
    # Image Upload
    st.divider()
    st.subheader("📎 Image Upload")
    uploaded_file = st.file_uploader(
        "Upload image (for vision models)",
        type=["png", "jpg", "jpeg", "gif", "webp"],
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        st.session_state.uploaded_image = uploaded_file
        st.session_state.uploaded_image_b64 = encode_image_to_base64(uploaded_file)
        st.session_state.uploaded_image_mime = safe_get_image_mime_type(uploaded_file)
        st.image(uploaded_file, width=150, caption=f"{uploaded_file.name} · {st.session_state.uploaded_image_mime}")
        if st.button("🗑️ Remove Image", use_container_width=True):
            st.session_state.uploaded_image = None
            st.session_state.uploaded_image_b64 = None
            st.session_state.uploaded_image_mime = None
            st.rerun()
    elif st.session_state.get("uploaded_image"):
        st.caption(f"Image ready · {st.session_state.get('uploaded_image_mime', 'image/jpeg')}")
    
    # API Keys Input
    user_api_keys: Dict[str, str] = dict(st.session_state.get("ad_hoc_api_keys", {}))
    missing_keys_info: List[Tuple[str, str]] = []
    
    for model_name in selected_models:
        key_name = all_models[model_name]["api_key_name"]
        has_session_key = bool(st.session_state.get("ad_hoc_api_keys", {}).get(model_name))
        if key_name not in st.secrets and not has_session_key:
            if not any(k == key_name for _, k in missing_keys_info):
                missing_keys_info.append((model_name, key_name))

    if missing_keys_info:
        st.divider()
        st.subheader("🔑 API Keys")
        for model_name, key_name in missing_keys_info:
            user_input = st.text_input(
                f"{key_name}", 
                type="password",
                key=f"key_input_{key_name}"
            )
            if user_input:
                for m_name in selected_models:
                    if all_models[m_name]["api_key_name"] == key_name:
                        user_api_keys[m_name] = user_input

    if selected_models:
        with st.expander("🧪 Model readiness", expanded=False):
            for model_name in selected_models:
                cfg = all_models[model_name]
                ready = model_has_api_key(model_name, user_api_keys)
                vision = "vision" if cfg.get("supports_vision") else "text-only"
                status = "🟢 ready" if ready else f"🔴 missing {cfg['api_key_name']}"
                st.caption(f"{cfg['icon']} **{model_name}** · {status} · {vision}")
                st.caption(f"↳ `{cfg.get('api_id')}` · `{cfg.get('provider')}`")
    
    with st.expander("ℹ️ Get API Keys"):
        st.markdown("""
        - **OpenAI:** [platform.openai.com](https://platform.openai.com/api-keys)
        - **Google:** [aistudio.google.com](https://aistudio.google.com/app/apikey)
        - **DeepSeek:** [platform.deepseek.com](https://platform.deepseek.com)
        - **xAI:** [console.x.ai](https://console.x.ai)
        """)

    # Session Stats
    st.divider()
    st.subheader("📊 Stats")
    col_cost, col_tok = st.columns(2)
    col_cost.metric("Estimated Cost", f"${st.session_state.session_cost:.4f}")
    col_tok.metric("Tokens", f"{st.session_state.total_tokens:,}")
    st.caption("Token/cost values are exact when providers return usage and estimated when streaming usage is unavailable.")
    with st.expander("📈 Model performance", expanded=False):
        perf_df = compute_model_performance(st.session_state.messages)
        if perf_df.empty:
            st.caption("No model calls yet.")
        else:
            st.dataframe(perf_df, use_container_width=True, hide_index=True)
    
    # Export
    st.divider()
    st.subheader("📥 Export Chat")
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        if st.session_state.messages:
            md_export = export_chat_markdown()
            st.download_button(
                "📄 Markdown",
                data=md_export,
                file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True
            )
        else:
            st.button("📄 Markdown", disabled=True, use_container_width=True)
    
    with col_exp2:
        if st.session_state.messages:
            json_export = export_chat_json()
            st.download_button(
                "📋 JSON",
                data=json_export,
                file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        else:
            st.button("📋 JSON", disabled=True, use_container_width=True)
    
    # Actions
    st.divider()
    col1, col2 = st.columns(2)
    
    if col1.button("🗑️ Clear", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_cost = 0.0
        st.session_state.total_tokens = 0
        st.session_state.response_batches = []
        st.session_state.last_batch_id = 0
        st.session_state.uploaded_image = None
        st.session_state.uploaded_image_b64 = None
        st.session_state.uploaded_image_mime = None
        st.rerun()
        
    if col2.button("🚪 Logout", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# --- 7. MAIN CHAT INTERFACE ---

def retry_message(message_id: int, agent_name: str):
    """Retry a response by truncating the conversation back to the context that produced it."""
    for i, msg in enumerate(st.session_state.messages):
        if msg.get("id") == message_id:
            st.session_state.messages = st.session_state.messages[:i]
            st.session_state["retry_agent"] = agent_name
            st.rerun()
    st.warning("Could not find the selected response to retry.")


# Create agents for selected models
active_agents: List[Agent] = []
for name in selected_models:
    user_key = user_api_keys.get(name)
    custom_prompt = st.session_state.custom_prompts.get(name)
    agent = Agent(
        name,
        get_model_config(name),
        user_key,
        concise_mode,
        temperature,
        custom_prompt,
        gemini_safety_mode,
        request_timeout
    )
    active_agents.append(agent)

# Latest comparison and synthesis tools
if st.session_state.get("response_batches"):
    latest_batch = st.session_state.response_batches[-1]
    latest_rows = summarize_response_rows(latest_batch)
    with st.expander("📊 Latest response comparison", expanded=False):
        st.caption(f"Prompt: {latest_batch.get('prompt', '')[:240]}")
        if latest_rows:
            st.dataframe(pd.DataFrame(latest_rows), use_container_width=True, hide_index=True)
        synth_candidates = [a.name for a in active_agents]
        if synth_candidates:
            synth_model = st.selectbox("Synthesis model", options=synth_candidates, key="synthesis_model_select")
            if st.button("✨ Synthesize latest responses", use_container_width=True):
                st.session_state["pending_synthesis"] = {"batch_id": latest_batch.get("batch_id"), "model": synth_model}
                st.rerun()

pending_synthesis = st.session_state.pop("pending_synthesis", None)
if pending_synthesis:
    target_batch = next((b for b in st.session_state.get("response_batches", []) if b.get("batch_id") == pending_synthesis.get("batch_id")), None)
    synth_agent = next((a for a in active_agents if a.name == pending_synthesis.get("model")), None)
    if target_batch and synth_agent:
        synthesis_text = "\n\n".join(f"### {r.get('name')}\n{r.get('content')}" for r in target_batch.get("responses", []) if not r.get("is_error"))
        synthesis_prompt = (
            "Create one best combined answer from the model responses below. Preserve useful nuance, correct weaknesses, "
            "avoid inventing facts, and be clear about uncertainty.\n\n" + synthesis_text
        )
        synth_history = get_limited_history(st.session_state.messages, history_limit) + [{
            "id": get_next_message_id(),
            "role": "user",
            "name": "User",
            "content": synthesis_prompt,
            "avatar": "👤"
        }]
        with st.chat_message("assistant", avatar=synth_agent.avatar):
            placeholder = st.empty()
            placeholder.markdown(f"**{synth_agent.name} Synthesis**: *combining responses...*")
            content, in_tok, out_tok, elapsed = synth_agent.generate_response_streaming(synth_history, placeholder, None, "image/jpeg")
            add_assistant_message(synth_agent, content, elapsed, in_tok, out_tok, is_judge=True, batch_id=target_batch.get("batch_id"))
            st.caption(f"⏱️ {elapsed:.1f}s • Synthesis")
        st.rerun()
    else:
        st.warning("Could not synthesize because the selected batch or model is no longer available.")

# Display chat history
for i, message in enumerate(st.session_state.messages):
    avatar = message.get("avatar", "👤" if message["role"] == "user" else "🤖")
    
    with st.chat_message(message["role"], avatar=avatar):
        col_msg, col_meta = st.columns([0.82, 0.18])
        
        with col_msg:
            label = "Judge" if message.get("is_judge") else message.get("name", "Unknown")
            st.markdown(f"**{label}**: {message.get('content', '')}")
            if message["role"] == "assistant" and not message.get("is_error"):
                with st.expander("📋 Copy response", expanded=False):
                    st.text_area(
                        "Response text",
                        value=message.get("content", ""),
                        height=160,
                        key=f"copy_{message.get('id', i)}",
                        label_visibility="collapsed"
                    )
            elif message["role"] == "assistant" and message.get("is_error"):
                st.caption("Failure details are available in provider dashboards and Streamlit logs.")
        
        with col_meta:
            if message["role"] == "assistant":
                if message.get("model_id"):
                    st.caption(f"🧭 `{message.get('model_id')}`")
                if message.get("response_time"):
                    st.caption(f"⏱️ {message['response_time']:.1f}s")
                if message.get("input_tokens") is not None and message.get("output_tokens") is not None:
                    st.caption(f"🔢 {message.get('input_tokens', 0):,}/{message.get('output_tokens', 0):,}")
                if message.get("estimated_cost") is not None:
                    st.caption(f"💵 ${message.get('estimated_cost', 0.0):.5f}")
                if message.get("is_debate"):
                    st.caption("🔄 Debate")
                if message.get("is_judge"):
                    st.caption("⚖️ Judge")
                if not message.get("is_error") and st.button("🔄", key=f"retry_hist_{message.get('id', i)}", help="Retry this response"):
                    retry_message(message.get("id"), message.get("name"))

# Check for pending retry
retry_agent_name = st.session_state.pop("retry_agent", None)

if retry_agent_name:
    agent = next((a for a in active_agents if a.name == retry_agent_name), None)
    if agent:
        with st.chat_message("assistant", avatar=agent.avatar):
            placeholder = st.empty()
            placeholder.markdown(f"**{agent.name}**: *retrying...*")
            
            content, in_tok, out_tok, elapsed = agent.generate_response_streaming(
                get_limited_history(st.session_state.messages, history_limit),
                placeholder,
                None,
                "image/jpeg"
            )
            
            msg_id = add_assistant_message(agent, content, elapsed, in_tok, out_tok)
            st.caption(f"⏱️ {elapsed:.1f}s (retry)")
        
        st.rerun()
    else:
        st.warning(f"Could not retry because {retry_agent_name} is not currently selected.")

# Process chat input
placeholder_text = "Type your message..." if active_agents else "← Select models first"

if user_input := st.chat_input(placeholder_text, disabled=not active_agents):
    
    if not active_agents:
        st.error("⚠️ Please select at least one AI model from the sidebar.")
        st.stop()

    # Apply template if selected
    if selected_template != "None":
        user_input = PROMPT_TEMPLATES[selected_template] + user_input

    # Add user message
    user_message = {
        "id": get_next_message_id(),
        "role": "user", 
        "name": "User", 
        "content": user_input, 
        "avatar": "👤"
    }
    st.session_state.messages.append(user_message)
    
    with st.chat_message("user", avatar="👤"):
        st.markdown(f"**User**: {user_input}")

    # Determine responders
    responders = parse_mentions(user_input, active_agents)
    
    # Get image if uploaded
    image_b64 = st.session_state.get("uploaded_image_b64")
    image_mime = st.session_state.get("uploaded_image_mime") or "image/jpeg"
    initial_response_records: List[Dict[str, Any]] = []
    batch_id = next_batch_id()

    # --- SIDE BY SIDE MODE ---
    if side_by_side and len(responders) > 1:
        cols = st.columns(len(responders))
        
        for idx, agent in enumerate(responders):
            with cols[idx]:
                with st.chat_message("assistant", avatar=agent.avatar):
                    placeholder = st.empty()
                    placeholder.markdown(f"**{agent.name}**: *thinking...*")
                    
                    content, in_tok, out_tok, elapsed = agent.generate_response_streaming(
                        get_limited_history(st.session_state.messages, history_limit),
                        placeholder,
                        image_b64 if agent.supports_vision else None,
                        image_mime
                    )
                    
                    msg_id = add_assistant_message(agent, content, elapsed, in_tok, out_tok, batch_id=batch_id)
                    initial_response_records.append({
                        "name": agent.name, "content": content, "model_id": agent.model_id,
                        "provider": agent.provider, "response_time": elapsed,
                        "input_tokens": in_tok, "output_tokens": out_tok,
                        "estimated_cost": calculate_cost(agent.name, in_tok, out_tok),
                        "is_error": is_error_content(content)
                    })
                    st.caption(f"⏱️ {elapsed:.1f}s")
    
    # --- STANDARD MODE ---
    else:
        for agent in responders:
            with st.chat_message("assistant", avatar=agent.avatar):
                placeholder = st.empty()
                placeholder.markdown(f"**{agent.name}**: *thinking...*")
                
                content, in_tok, out_tok, elapsed = agent.generate_response_streaming(
                    get_limited_history(st.session_state.messages, history_limit),
                    placeholder,
                    image_b64 if agent.supports_vision else None,
                    image_mime
                )
                
                msg_id = add_assistant_message(agent, content, elapsed, in_tok, out_tok, batch_id=batch_id)
                initial_response_records.append({
                    "name": agent.name, "content": content, "model_id": agent.model_id,
                    "provider": agent.provider, "response_time": elapsed,
                    "input_tokens": in_tok, "output_tokens": out_tok,
                    "estimated_cost": calculate_cost(agent.name, in_tok, out_tok),
                    "is_error": is_error_content(content)
                })
                
                col_time, col_retry = st.columns([0.8, 0.2])
                with col_time:
                    st.caption(f"⏱️ {elapsed:.1f}s")
                with col_retry:
                    if st.button("🔄", key=f"retry_{msg_id}", help="Retry this response"):
                        retry_message(msg_id, agent.name)
    
    if initial_response_records:
        st.session_state.response_batches.append({
            "batch_id": batch_id,
            "created_at": datetime.now().isoformat(),
            "prompt": user_input,
            "responders": [r.get("name") for r in initial_response_records],
            "responses": initial_response_records,
        })
        # Keep session state modest on Streamlit Community Cloud.
        st.session_state.response_batches = st.session_state.response_batches[-25:]

    # --- DEBATE MODE ---
    if debate_mode and len(responders) > 1:
        st.divider()
        st.markdown("### 🔄 Debate Round")
        
        for agent in responders:
            with st.chat_message("assistant", avatar=agent.avatar):
                placeholder = st.empty()
                placeholder.markdown(f"**{agent.name}** *(responding to others)*: *thinking...*")
                
                content, in_tok, out_tok, elapsed = agent.generate_response_streaming(
                    get_limited_history(st.session_state.messages, history_limit),
                    placeholder,
                    None,
                    "image/jpeg"
                )
                
                add_assistant_message(agent, content, elapsed, in_tok, out_tok, is_debate=True, batch_id=batch_id)
                st.caption(f"⏱️ {elapsed:.1f}s • Debate Round")

    # --- JUDGE MODE ---
    if judge_mode and judge_model_name and len(initial_response_records) > 1:
        judge_agent = next((a for a in active_agents if a.name == judge_model_name), None)
        if judge_agent:
            st.divider()
            st.markdown("### ⚖️ Judge Assessment")
            comparison_text = "\n\n".join(
                f"### {item['name']}\n{item['content']}" for item in initial_response_records
            )
            judge_prompt = (
                "Compare the model responses below. Identify the strongest answer, factual risks, "
                "missing considerations, and a best combined answer. Be concise and practical.\n\n"
                f"{comparison_text}"
            )
            judge_history = get_limited_history(st.session_state.messages, history_limit) + [{
                "id": get_next_message_id(),
                "role": "user",
                "name": "User",
                "content": judge_prompt,
                "avatar": "👤"
            }]
            with st.chat_message("assistant", avatar=judge_agent.avatar):
                placeholder = st.empty()
                placeholder.markdown(f"**{judge_agent.name} Judge**: *reviewing responses...*")
                content, in_tok, out_tok, elapsed = judge_agent.generate_response_streaming(
                    judge_history,
                    placeholder,
                    None,
                    "image/jpeg"
                )
                add_assistant_message(judge_agent, content, elapsed, in_tok, out_tok, is_judge=True, batch_id=batch_id)
                st.caption(f"⏱️ {elapsed:.1f}s • Judge Mode")
    
    # Clear image after use
    if image_b64:
        st.session_state.uploaded_image = None
        st.session_state.uploaded_image_b64 = None
        st.session_state.uploaded_image_mime = None
    
    st.rerun()
