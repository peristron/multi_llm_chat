import streamlit as st
import hmac
import re
from typing import Optional, Tuple, List, Dict, Any
from openai import OpenAI
import google.generativeai as genai

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
        "system_prompt": "You are GPT-4o, a helpful AI assistant made by OpenAI. You are helpful, academic, and structured.",
        "mention_triggers": ["gpt", "gpt4", "gpt4o", "openai"]
    },
    "Grok-3": {
        "api_id": "grok-3",
        "provider": "openai_compatible",
        "base_url": "https://api.x.ai/v1",
        "api_key_name": "XAI_API_KEY",
        "price_input": 3.00,
        "price_output": 15.00,
        "icon": "⚫",
        "system_prompt": "You are Grok 3, made by xAI. You are witty, direct, and enjoy intellectual discourse.",
        "mention_triggers": ["grok", "grok3", "xai"]
    },
    "Gemini 2.5 Flash": {
        "api_id": "gemini-2.5-flash-preview-05-20",
        "provider": "google",
        "api_key_name": "GOOGLE_API_KEY",
        "price_input": 0.075,
        "price_output": 0.30,
        "icon": "🔵",
        "system_prompt": "You are Gemini 2.5 Flash, made by Google. You are fast, efficient, and detailed.",
        "mention_triggers": ["gemini", "google", "flash"]
    },
    "DeepSeek V3": {
        "api_id": "deepseek-chat",
        "provider": "openai_compatible",
        "base_url": "https://api.deepseek.com",
        "api_key_name": "DEEPSEEK_API_KEY",
        "price_input": 0.14,
        "price_output": 0.28,
        "icon": "🦈",
        "system_prompt": "You are DeepSeek V3, a highly capable AI assistant. You are analytical and precise.",
        "mention_triggers": ["deepseek", "ds", "deep"]
    }
}

# --- 2. SESSION STATE INITIALIZATION (Early) ---
def init_session_state():
    defaults = {
        "messages": [],
        "session_cost": 0.0,
        "total_tokens": 0,
        "password_correct": False
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
            st.session_state.password_correct = True
        else:
            st.session_state.password_correct = False
        # Clear password from state
        if "password" in st.session_state:
            del st.session_state["password"]

    if st.session_state.password_correct:
        return True

    st.title("🔒 Login Required")
    st.text_input(
        "Password:", 
        type="password", 
        on_change=password_entered, 
        key="password"
    )
    
    if "password_correct" in st.session_state and st.session_state.password_correct is False:
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
    - Example: `@grok what do you think about this?`
    
    **Tips:**
    - Conversation history persists - models remember previous context
    - Use "Concise Responses" toggle for shorter answers
    - Session costs and token usage are tracked in the sidebar
    """)

# --- 4. HELPER FUNCTIONS ---

@st.cache_resource
def get_openai_client(api_key: str, base_url: Optional[str] = None) -> OpenAI:
    """Cache OpenAI-compatible clients to avoid recreation."""
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

def parse_mentions(user_input: str, active_agents: List['Agent']) -> List['Agent']:
    """Parse @mentions to determine which agents should respond."""
    lower_input = user_input.lower()
    mentioned_agents = []
    
    for agent in active_agents:
        triggers = agent.config.get("mention_triggers", [])
        # Check each trigger
        for trigger in triggers:
            if f"@{trigger}" in lower_input:
                if agent not in mentioned_agents:
                    mentioned_agents.append(agent)
                break
    
    # If no specific mentions, all agents respond
    return mentioned_agents if mentioned_agents else active_agents

# --- 5. AGENT CLASS ---
class Agent:
    # Known AI names for identity filtering
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
        concise_mode: bool = True
    ):
        self.name = display_name
        self.config = config
        self.model_id = config["api_id"]
        self.provider = config["provider"]
        self.avatar = config["icon"]
        
        # Build system prompt
        base_prompt = config["system_prompt"]
        if concise_mode:
            base_prompt += (
                "\n\nIMPORTANT INSTRUCTIONS:"
                "\n- Be brief, concise, and direct in your responses."
                "\n- Do not prefix your response with your name."
                "\n- Do not roleplay as or speak for other AI models."
                "\n- Only provide detailed explanations when explicitly requested."
            )
        self.system_prompt = base_prompt
        
        # Validate API key availability
        self.api_key = get_api_key(config["api_key_name"], user_key)
        self.error = None if self.api_key else f"Missing API Key: {config['api_key_name']}"

    def _clean_response(self, text: str) -> str:
        """Clean model response of identity headers and hallucinated content."""
        if not text or not text.strip():
            return "..."
        
        text = text.strip()
        
        # Build regex pattern from known names (escaped for safety)
        escaped_names = [re.escape(name) for name in self.KNOWN_AI_NAMES]
        escaped_names.extend([re.escape(name) for name in AVAILABLE_MODELS.keys()])
        names_pattern = "|".join(escaped_names)
        
        # 1. Remove identity header at start: "ModelName:" or "ModelName -" or "[ModelName]:"
        patterns_to_remove = [
            rf"^\[?({names_pattern})\]?\s*[:\-]\s*",  # Name: or [Name]:
            rf"^\*\*({names_pattern})\*\*\s*[:\-]?\s*",  # **Name**: 
        ]
        for pattern in patterns_to_remove:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
        
        # 2. Cut off hallucinated continuations (other models "speaking")
        continuation_pattern = rf"\n\s*\[?({names_pattern})\]?\s*[:\-]"
        match = re.search(continuation_pattern, text, flags=re.IGNORECASE)
        if match:
            text = text[:match.start()].strip()
        
        # 3. Also catch markdown-style continuations
        md_continuation = rf"\n\s*\*\*({names_pattern})\*\*\s*[:\-]?"
        match = re.search(md_continuation, text, flags=re.IGNORECASE)
        if match:
            text = text[:match.start()].strip()
        
        return text if text else "..."

    def generate_response(self, conversation_history: List[Dict]) -> Tuple[str, int, int]:
        """Generate a response based on conversation history."""
        if self.error:
            return f"⚠️ {self.error}", 0, 0

        try:
            if self.provider == "google":
                content, in_tok, out_tok = self._call_google(conversation_history)
            else:
                content, in_tok, out_tok = self._call_openai_compatible(conversation_history)
            
            return self._clean_response(content), in_tok, out_tok
        except Exception as e:
            return f"⚠️ Unexpected error: {str(e)}", 0, 0

    def _format_history_message(self, msg: Dict) -> str:
        """Format a history message with clear speaker attribution."""
        speaker = msg.get('name', 'Unknown')
        content = msg.get('content', '')
        # Use brackets to clearly delineate speakers
        return f"[{speaker}]: {content}"

    def _call_openai_compatible(self, history: List[Dict]) -> Tuple[str, int, int]:
        """Call OpenAI-compatible API (OpenAI, Grok, DeepSeek)."""
        messages = [{"role": "system", "content": self.system_prompt}]
        
        for msg in history:
            role = "user" if msg["role"] == "user" else "assistant"
            content = self._format_history_message(msg)
            messages.append({"role": role, "content": content})

        try:
            client = get_openai_client(self.api_key, self.config.get("base_url"))
            
            response = client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=0.7,
                max_tokens=2048
            )
            
            content = response.choices[0].message.content or ""
            usage = response.usage
            in_tok = usage.prompt_tokens if usage else 0
            out_tok = usage.completion_tokens if usage else 0
            return content, in_tok, out_tok
            
        except Exception as e:
            error_msg = str(e).lower()
            if "rate" in error_msg and "limit" in error_msg:
                return "⚠️ Rate limit reached. Please wait a moment and try again.", 0, 0
            elif "invalid" in error_msg and "key" in error_msg:
                return "⚠️ Invalid API key. Please check your credentials.", 0, 0
            elif "insufficient" in error_msg and "quota" in error_msg:
                return "⚠️ API quota exceeded. Please check your account.", 0, 0
            return f"⚠️ API Error: {str(e)}", 0, 0

    def _call_google(self, history: List[Dict]) -> Tuple[str, int, int]:
        """Call Google Gemini API with automatic fallback."""
        
        def execute_gemini(model_name: str) -> Tuple[str, int, int]:
            # Configure API for this request
            genai.configure(api_key=self.api_key)
            
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=2048,
                temperature=0.7
            )
            
            model = genai.GenerativeModel(
                model_name,
                system_instruction=self.system_prompt,
                generation_config=generation_config
            )
            
            # Convert history to Google format
            google_history = []
            for msg in history:
                role = "user" if msg["role"] == "user" else "model"
                content = self._format_history_message(msg)
                google_history.append({"role": role, "parts": [content]})

            # Safety settings as dict
            safety_settings = {
                "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
            }
            
            response = model.generate_content(
                google_history, 
                safety_settings=safety_settings
            )
            
            # Handle blocked responses
            try:
                text = response.text
            except ValueError:
                # Check if response was blocked
                if hasattr(response, 'prompt_feedback'):
                    return "Response blocked by safety filters.", 0, 0
                return "Unable to generate response.", 0, 0

            usage = response.usage_metadata
            in_tok = getattr(usage, 'prompt_token_count', 0)
            out_tok = getattr(usage, 'candidates_token_count', 0)
            return text, in_tok, out_tok

        # Try primary model first
        try:
            return execute_gemini(self.model_id)
        except Exception as e:
            error_str = str(e).lower()
            
            # Only attempt fallback for "not found" errors
            if "404" in error_str or "not found" in error_str or "not supported" in error_str:
                return self._google_fallback(execute_gemini, str(e))
            
            # Handle other specific errors
            if "rate" in error_str and "limit" in error_str:
                return "⚠️ Rate limit reached. Please wait and try again.", 0, 0
            elif "quota" in error_str:
                return "⚠️ API quota exceeded.", 0, 0
            elif "api key" in error_str:
                return "⚠️ Invalid Google API key.", 0, 0
                
            return f"⚠️ Google API Error: {str(e)}", 0, 0

    def _google_fallback(self, execute_fn, original_error: str) -> Tuple[str, int, int]:
        """Attempt to use fallback Gemini models."""
        fallback_models = [
            "gemini-2.5-flash-preview-05-20",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-1.5-flash",
            "gemini-1.5-flash-latest",
            "gemini-1.5-pro"
        ]
        
        # Remove the model that just failed
        fallback_models = [m for m in fallback_models if m != self.model_id]
        
        for fallback_model in fallback_models:
            try:
                text, in_tok, out_tok = execute_fn(fallback_model)
                if text and not text.startswith("⚠️"):
                    note = f"\n\n*(Auto-switched from {self.model_id} to {fallback_model})*"
                    return text + note, in_tok, out_tok
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
        help="Choose which AI models will participate in the chat"
    )
    
    if not selected_models:
        st.warning("Please select at least one model.")

    # Settings
    st.caption("Settings")
    concise_mode = st.checkbox(
        "Concise Responses", 
        value=True, 
        help="Instructs all models to keep answers brief and direct."
    )

    # Dynamic API key input for missing keys
    user_api_keys: Dict[str, str] = {}
    missing_keys_info: List[Tuple[str, str]] = []
    
    for model_name in selected_models:
        key_name = AVAILABLE_MODELS[model_name]["api_key_name"]
        if key_name not in st.secrets:
            # Check if we already have this key_name (shared keys)
            if not any(k == key_name for _, k in missing_keys_info):
                missing_keys_info.append((model_name, key_name))

    if missing_keys_info:
        st.divider()
        st.caption("🔑 API Keys Required")
        
        for model_name, key_name in missing_keys_info:
            user_input = st.text_input(
                f"{key_name}", 
                type="password",
                key=f"key_input_{key_name}",
                help=f"Required for {model_name}"
            )
            if user_input:
                # Apply to all models using this key
                for m_name in selected_models:
                    if AVAILABLE_MODELS[m_name]["api_key_name"] == key_name:
                        user_api_keys[m_name] = user_input
    
    with st.expander("ℹ️ Get API Keys"):
        st.markdown("""
        - **OpenAI:** [platform.openai.com](https://platform.openai.com/api-keys)
        - **Google:** [aistudio.google.com](https://aistudio.google.com/app/apikey)
        - **DeepSeek:** [platform.deepseek.com](https://platform.deepseek.com)
        - **xAI (Grok):** [console.x.ai](https://console.x.ai)
        """)

    # Session Stats
    st.divider()
    st.header("📊 Session Stats")
    col_cost, col_tok = st.columns(2)
    col_cost.metric("Cost", f"${st.session_state.session_cost:.4f}")
    col_tok.metric("Tokens", f"{st.session_state.total_tokens:,}")
    
    # Actions
    st.divider()
    col1, col2 = st.columns(2)
    
    if col1.button("🗑️ Clear", use_container_width=True, help="Clear chat history"):
        st.session_state.messages = []
        st.session_state.session_cost = 0.0
        st.session_state.total_tokens = 0
        st.rerun()
        
    if col2.button("🚪 Logout", use_container_width=True, help="Log out of the app"):
        st.session_state.password_correct = False
        st.session_state.messages = []
        st.session_state.session_cost = 0.0
        st.session_state.total_tokens = 0
        st.rerun()


# --- 7. MAIN CHAT INTERFACE ---

# Create agents for selected models
active_agents: List[Agent] = []
for name in selected_models:
    user_key = user_api_keys.get(name)
    agent = Agent(name, AVAILABLE_MODELS[name], user_key, concise_mode)
    active_agents.append(agent)

# Display chat history
for message in st.session_state.messages:
    avatar = message.get("avatar", "👤" if message["role"] == "user" else "🤖")
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(f"**{message['name']}**: {message['content']}")

# Chat input placeholder changes based on state
placeholder_text = "Type your message..." if active_agents else "← Select models first"

# Process chat input
if user_input := st.chat_input(placeholder_text):
    
    if not active_agents:
        st.error("Please select at least one AI model from the sidebar.")
        st.stop()

    # Add and display user message
    user_message = {
        "role": "user", 
        "name": "User", 
        "content": user_input, 
        "avatar": "👤"
    }
    st.session_state.messages.append(user_message)
    
    with st.chat_message("user", avatar="👤"):
        st.markdown(f"**User**: {user_input}")

    # Determine which agents should respond based on @mentions
    responders = parse_mentions(user_input, active_agents)

    # Generate and display responses from each responding agent
    for agent in responders:
        with st.chat_message("assistant", avatar=agent.avatar):
            with st.spinner(f"{agent.name} is thinking..."):
                content, in_tok, out_tok = agent.generate_response(st.session_state.messages)
            
            st.markdown(f"**{agent.name}**: {content}")
            
            # Save response to history
            st.session_state.messages.append({
                "role": "assistant", 
                "name": agent.name, 
                "content": content, 
                "avatar": agent.avatar
            })
            
            # Update session statistics
            cost = calculate_cost(agent.name, in_tok, out_tok)
            st.session_state.session_cost += cost
            st.session_state.total_tokens += (in_tok + out_tok)
    
    # Rerun to update sidebar stats and ensure clean state
    st.rerun()
