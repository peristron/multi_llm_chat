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

# --- 2. SESSION STATE INITIALIZATION ---
def init_session_state():
    defaults = {
        "messages": [],
        "session_cost": 0.0,
        "total_tokens": 0,
        "custom_prompts": {name: config["default_system_prompt"] for name, config in AVAILABLE_MODELS.items()},
        "message_id_counter": 0,
        "uploaded_image": None,
        "uploaded_image_b64": None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# --- 3. AUTHENTICATION ---
def check_password() -> bool:
    def password_entered():
        entered = st.session_state.get("password", "")
        correct = st.secrets.get("APP_PASSWORD", "")
        if entered and correct and hmac.compare_digest(entered, correct):
            st.session_state["password_correct"] = True
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
def get_openai_client(api_key: str, base_url: Optional[str] = None) -> OpenAI:
    """Cache OpenAI-compatible clients."""
    return OpenAI(api_key=api_key, base_url=base_url)

def get_api_key(key_name: str, user_provided_key: Optional[str] = None) -> Optional[str]:
    """Get API key from user input or secrets."""
    if user_provided_key:
        return user_provided_key
    return st.secrets.get(key_name)

def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate API cost based on token usage."""
    if model_name not in AVAILABLE_MODELS:
        return 0.0
    info = AVAILABLE_MODELS[model_name]
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
    """Export chat history as Markdown."""
    lines = ["# Chat Export", f"*Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*", ""]
    
    for msg in st.session_state.messages:
        name = msg.get("name", "Unknown")
        content = msg.get("content", "")
        role_icon = "👤" if msg["role"] == "user" else msg.get("avatar", "🤖")
        lines.append(f"### {role_icon} {name}")
        lines.append(content)
        if msg.get("response_time"):
            lines.append(f"*Response time: {msg['response_time']:.2f}s*")
        lines.append("")
    
    lines.append("---")
    lines.append(f"**Total Cost:** ${st.session_state.session_cost:.5f}")
    lines.append(f"**Total Tokens:** {st.session_state.total_tokens:,}")
    
    return "\n".join(lines)

def export_chat_json() -> str:
    """Export chat history as JSON."""
    export_data = {
        "exported_at": datetime.now().isoformat(),
        "session_cost": st.session_state.session_cost,
        "total_tokens": st.session_state.total_tokens,
        "messages": st.session_state.messages
    }
    return json.dumps(export_data, indent=2)

def encode_image_to_base64(uploaded_file) -> Optional[str]:
    """Encode uploaded image to base64."""
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        return base64.standard_b64encode(bytes_data).decode("utf-8")
    return None

# --- 5. AGENT CLASS ---
class Agent:
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
        custom_system_prompt: Optional[str] = None
    ):
        self.name = display_name
        self.config = config
        self.model_id = config["api_id"]
        self.provider = config["provider"]
        self.avatar = config["icon"]
        self.temperature = temperature
        self.supports_vision = config.get("supports_vision", False)
        
        # Use custom prompt if provided, else default
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
        
        self.api_key = get_api_key(config["api_key_name"], user_key)
        self.error = None if self.api_key else f"Missing API Key: {config['api_key_name']}"

    def _clean_response(self, text: str) -> str:
        """Clean model response of identity headers."""
        if not text or not text.strip():
            return "..."
        
        text = text.strip()
        
        escaped_names = [re.escape(name) for name in self.KNOWN_AI_NAMES]
        escaped_names.extend([re.escape(name) for name in AVAILABLE_MODELS.keys()])
        names_pattern = "|".join(escaped_names)
        
        patterns_to_remove = [
            rf"^\[?({names_pattern})\]?\s*[:\-]\s*",
            rf"^\*\*({names_pattern})\*\*\s*[:\-]?\s*",
        ]
        for pattern in patterns_to_remove:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
        
        continuation_pattern = rf"\n\s*\[?({names_pattern})\]?\s*[:\-]"
        match = re.search(continuation_pattern, text, flags=re.IGNORECASE)
        if match:
            text = text[:match.start()].strip()
        
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
        image_b64: Optional[str] = None
    ) -> Tuple[str, int, int, float]:
        """Generate streaming response. Returns (content, in_tokens, out_tokens, time)."""
        
        if self.error:
            return f"⚠️ {self.error}", 0, 0, 0.0

        start_time = time.time()
        
        try:
            if self.provider == "google":
                content, in_tok, out_tok = self._stream_google(conversation_history, placeholder, image_b64)
            else:
                content, in_tok, out_tok = self._stream_openai_compatible(conversation_history, placeholder, image_b64)
            
            elapsed = time.time() - start_time
            return self._clean_response(content), in_tok, out_tok, elapsed
            
        except Exception as e:
            elapsed = time.time() - start_time
            return f"⚠️ Error: {str(e)}", 0, 0, elapsed

    def _stream_openai_compatible(
        self, 
        history: List[Dict], 
        placeholder,
        image_b64: Optional[str] = None
    ) -> Tuple[str, int, int]:
        """Stream from OpenAI-compatible API."""
        
        messages = [{"role": "system", "content": self.system_prompt}]
        
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
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]

        try:
            client = get_openai_client(self.api_key, self.config.get("base_url"))
            
            stream = client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=self.temperature,
                max_tokens=2048,
                stream=True
            )
            
            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    placeholder.markdown(f"**{self.name}**: {full_response}▌")
            
            placeholder.markdown(f"**{self.name}**: {full_response}")
            
            # Estimate tokens (streaming doesn't return usage)
            in_tok = sum(len(str(m.get("content", ""))) // 4 for m in messages)
            out_tok = len(full_response) // 4
            
            return full_response, in_tok, out_tok
            
        except Exception as e:
            error_msg = str(e).lower()
            if "rate" in error_msg and "limit" in error_msg:
                return "⚠️ Rate limit reached. Please wait.", 0, 0
            elif "invalid" in error_msg and "key" in error_msg:
                return "⚠️ Invalid API key.", 0, 0
            return f"⚠️ API Error: {str(e)}", 0, 0

    def _stream_google(
        self, 
        history: List[Dict], 
        placeholder,
        image_b64: Optional[str] = None
    ) -> Tuple[str, int, int]:
        """Stream from Google Gemini API."""
        
        genai.configure(api_key=self.api_key)
        
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=2048,
            temperature=self.temperature
        )
        
        model = genai.GenerativeModel(
            self.model_id,
            system_instruction=self.system_prompt,
            generation_config=generation_config
        )
        
        google_history = []
        for msg in history:
            role = "user" if msg["role"] == "user" else "model"
            content = self._format_history_message(msg)
            google_history.append({"role": role, "parts": [content]})

        # Add image to last user message if provided
        if image_b64 and self.supports_vision and google_history:
            for i in range(len(google_history) - 1, -1, -1):
                if google_history[i]["role"] == "user":
                    import base64 as b64module
                    image_bytes = b64module.b64decode(image_b64)
                    google_history[i]["parts"].append({
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_b64
                        }
                    })
                    break

        safety_settings = {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
        }
        
        try:
            response = model.generate_content(
                google_history,
                safety_settings=safety_settings,
                stream=True
            )
            
            full_response = ""
            for chunk in response:
                if chunk.text:
                    full_response += chunk.text
                    placeholder.markdown(f"**{self.name}**: {full_response}▌")
            
            placeholder.markdown(f"**{self.name}**: {full_response}")
            
            # Get token counts from final response
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
            if "404" in error_str or "not found" in error_str:
                return self._google_fallback_stream(history, placeholder, str(e))
            return f"⚠️ Google API Error: {str(e)}", 0, 0

    def _google_fallback_stream(
        self, 
        history: List[Dict], 
        placeholder, 
        original_error: str
    ) -> Tuple[str, int, int]:
        """Fallback to alternative Gemini models."""
        
        fallback_models = [
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gemini-1.5-flash-latest",
            "gemini-1.5-pro"
        ]
        
        fallback_models = [m for m in fallback_models if m != self.model_id]
        
        for fallback_model in fallback_models:
            try:
                placeholder.markdown(f"*Trying {fallback_model}...*")
                
                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel(
                    fallback_model,
                    system_instruction=self.system_prompt
                )
                
                google_history = []
                for msg in history:
                    role = "user" if msg["role"] == "user" else "model"
                    content = self._format_history_message(msg)
                    google_history.append({"role": role, "parts": [content]})

                safety_settings = {
                    "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
                }
                
                response = model.generate_content(
                    google_history,
                    safety_settings=safety_settings,
                    stream=True
                )
                
                full_response = ""
                for chunk in response:
                    if chunk.text:
                        full_response += chunk.text
                        placeholder.markdown(f"**{self.name}**: {full_response}▌")
                
                full_response += f"\n\n*(Switched to {fallback_model})*"
                placeholder.markdown(f"**{self.name}**: {full_response}")
                
                in_tok = len(str(google_history)) // 4
                out_tok = len(full_response) // 4
                return full_response, in_tok, out_tok
                
            except Exception:
                continue
        
        return f"⚠️ All Gemini models unavailable. Error: {original_error}", 0, 0


# --- 6. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection
    selected_models = st.multiselect(
        "Select Participants:",
        options=list(AVAILABLE_MODELS.keys()),
        default=["GPT-4o", "Grok-3"],
        help="Choose which AI models will participate"
    )
    
    if not selected_models:
        st.warning("Please select at least one model.")

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
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="Higher = more creative, Lower = more focused"
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
        for model_name in selected_models:
            new_prompt = st.text_area(
                f"{AVAILABLE_MODELS[model_name]['icon']} {model_name}",
                value=st.session_state.custom_prompts.get(model_name, AVAILABLE_MODELS[model_name]["default_system_prompt"]),
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
        st.image(uploaded_file, width=150)
        if st.button("🗑️ Remove Image", use_container_width=True):
            st.session_state.uploaded_image = None
            st.session_state.uploaded_image_b64 = None
            st.rerun()
    
    # API Keys Input
    user_api_keys: Dict[str, str] = {}
    missing_keys_info: List[Tuple[str, str]] = []
    
    for model_name in selected_models:
        key_name = AVAILABLE_MODELS[model_name]["api_key_name"]
        if key_name not in st.secrets:
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
                    if AVAILABLE_MODELS[m_name]["api_key_name"] == key_name:
                        user_api_keys[m_name] = user_input
    
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
    col_cost.metric("Cost", f"${st.session_state.session_cost:.4f}")
    col_tok.metric("Tokens", f"{st.session_state.total_tokens:,}")
    
    # Export
    st.divider()
    st.subheader("📥 Export Chat")
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        md_export = export_chat_markdown()
        st.download_button(
            "📄 Markdown",
            data=md_export,
            file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True
        )
    
    with col_exp2:
        json_export = export_chat_json()
        st.download_button(
            "📋 JSON",
            data=json_export,
            file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    # Actions
    st.divider()
    col1, col2 = st.columns(2)
    
    if col1.button("🗑️ Clear", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_cost = 0.0
        st.session_state.total_tokens = 0
        st.session_state.uploaded_image = None
        st.session_state.uploaded_image_b64 = None
        st.rerun()
        
    if col2.button("🚪 Logout", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# --- 7. MAIN CHAT INTERFACE ---

# Create agents for selected models
active_agents: List[Agent] = []
for name in selected_models:
    user_key = user_api_keys.get(name)
    custom_prompt = st.session_state.custom_prompts.get(name)
    agent = Agent(
        name, 
        AVAILABLE_MODELS[name], 
        user_key, 
        concise_mode,
        temperature,
        custom_prompt
    )
    active_agents.append(agent)

# Display chat history
for i, message in enumerate(st.session_state.messages):
    avatar = message.get("avatar", "👤" if message["role"] == "user" else "🤖")
    
    with st.chat_message(message["role"], avatar=avatar):
        col_msg, col_meta = st.columns([0.9, 0.1])
        
        with col_msg:
            st.markdown(f"**{message['name']}**: {message['content']}")
        
        with col_meta:
            # Show response time for assistant messages
            if message["role"] == "assistant" and message.get("response_time"):
                st.caption(f"⏱️ {message['response_time']:.1f}s")

# --- RETRY FUNCTION ---
def retry_message(message_id: int, agent_name: str):
    """Retry generating a response for a specific message."""
    # Find and remove the message to retry
    for i, msg in enumerate(st.session_state.messages):
        if msg.get("id") == message_id:
            st.session_state.messages.pop(i)
            break
    st.session_state["retry_agent"] = agent_name
    st.rerun()

# Check for pending retry
retry_agent_name = st.session_state.pop("retry_agent", None)

# Process chat input
placeholder_text = "Type your message..." if active_agents else "← Select models first"

if user_input := st.chat_input(placeholder_text):
    
    if not active_agents:
        st.error("Please select at least one AI model from the sidebar.")
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

    # --- SIDE BY SIDE MODE ---
    if side_by_side and len(responders) > 1:
        cols = st.columns(len(responders))
        
        for idx, agent in enumerate(responders):
            with cols[idx]:
                with st.chat_message("assistant", avatar=agent.avatar):
                    placeholder = st.empty()
                    placeholder.markdown(f"**{agent.name}**: *thinking...*")
                    
                    content, in_tok, out_tok, elapsed = agent.generate_response_streaming(
                        st.session_state.messages,
                        placeholder,
                        image_b64 if agent.supports_vision else None
                    )
                    
                    msg_id = get_next_message_id()
                    st.session_state.messages.append({
                        "id": msg_id,
                        "role": "assistant", 
                        "name": agent.name, 
                        "content": content, 
                        "avatar": agent.avatar,
                        "response_time": elapsed
                    })
                    
                    st.caption(f"⏱️ {elapsed:.1f}s")
                    
                    cost = calculate_cost(agent.name, in_tok, out_tok)
                    st.session_state.session_cost += cost
                    st.session_state.total_tokens += (in_tok + out_tok)
    
    # --- STANDARD MODE ---
    else:
        for agent in responders:
            with st.chat_message("assistant", avatar=agent.avatar):
                placeholder = st.empty()
                placeholder.markdown(f"**{agent.name}**: *thinking...*")
                
                content, in_tok, out_tok, elapsed = agent.generate_response_streaming(
                    st.session_state.messages,
                    placeholder,
                    image_b64 if agent.supports_vision else None
                )
                
                msg_id = get_next_message_id()
                st.session_state.messages.append({
                    "id": msg_id,
                    "role": "assistant", 
                    "name": agent.name, 
                    "content": content, 
                    "avatar": agent.avatar,
                    "response_time": elapsed
                })
                
                # Show timing and retry button
                col_time, col_retry = st.columns([0.8, 0.2])
                with col_time:
                    st.caption(f"⏱️ {elapsed:.1f}s")
                with col_retry:
                    if st.button("🔄", key=f"retry_{msg_id}", help="Retry"):
                        retry_message(msg_id, agent.name)
                
                cost = calculate_cost(agent.name, in_tok, out_tok)
                st.session_state.session_cost += cost
                st.session_state.total_tokens += (in_tok + out_tok)
    
    # --- DEBATE MODE ---
    if debate_mode and len(responders) > 1:
        st.divider()
        st.markdown("### 🔄 Debate Round")
        
        # Each model responds to the previous responses
        for agent in responders:
            with st.chat_message("assistant", avatar=agent.avatar):
                placeholder = st.empty()
                placeholder.markdown(f"**{agent.name}** *(responding to others)*: *thinking...*")
                
                content, in_tok, out_tok, elapsed = agent.generate_response_streaming(
                    st.session_state.messages,
                    placeholder,
                    None  # No image in debate round
                )
                
                msg_id = get_next_message_id()
                st.session_state.messages.append({
                    "id": msg_id,
                    "role": "assistant", 
                    "name": agent.name, 
                    "content": content, 
                    "avatar": agent.avatar,
                    "response_time": elapsed,
                    "is_debate": True
                })
                
                st.caption(f"⏱️ {elapsed:.1f}s")
                
                cost = calculate_cost(agent.name, in_tok, out_tok)
                st.session_state.session_cost += cost
                st.session_state.total_tokens += (in_tok + out_tok)
    
    # Clear image after use
    if image_b64:
        st.session_state.uploaded_image = None
        st.session_state.uploaded_image_b64 = None
    
    st.rerun()

# Handle retry if pending
if retry_agent_name:
    agent = next((a for a in active_agents if a.name == retry_agent_name), None)
    if agent:
        with st.chat_message("assistant", avatar=agent.avatar):
            placeholder = st.empty()
            placeholder.markdown(f"**{agent.name}**: *retrying...*")
            
            content, in_tok, out_tok, elapsed = agent.generate_response_streaming(
                st.session_state.messages,
                placeholder,
                None
            )
            
            msg_id = get_next_message_id()
            st.session_state.messages.append({
                "id": msg_id,
                "role": "assistant", 
                "name": agent.name, 
                "content": content, 
                "avatar": agent.avatar,
                "response_time": elapsed
            })
            
            st.caption(f"⏱️ {elapsed:.1f}s (retry)")
            
            cost = calculate_cost(agent.name, in_tok, out_tok)
            st.session_state.session_cost += cost
            st.session_state.total_tokens += (in_tok + out_tok)
        
        st.rerun()
