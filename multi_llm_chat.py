import streamlit as st
import hmac
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
        "system_prompt": "You are GPT-4o. You are helpful, academic, and structured."
    },
    "Grok-3": {
        "api_id": "grok-3",
        "provider": "openai_compatible",
        "base_url": "https://api.x.ai/v1",
        "api_key_name": "XAI_API_KEY",
        "price_input": 3.00,
        "price_output": 15.00,
        "icon": "⚫",
        "system_prompt": "You are Grok 3. You are witty, rebellious, and enjoy debating."
    },
    "Gemini 1.5 Pro": {
        "api_id": "gemini-1.5-pro",
        "provider": "google",
        "api_key_name": "GOOGLE_API_KEY",
        "price_input": 1.25,
        "price_output": 5.00,
        "icon": "🔵",
        "system_prompt": "You are Gemini 1.5 Pro. You are detailed and nuanced."
    },
    "DeepSeek V3": {
        "api_id": "deepseek-chat",
        "provider": "openai_compatible",
        "base_url": "https://api.deepseek.com",
        "api_key_name": "DEEPSEEK_API_KEY",
        "price_input": 0.14,
        "price_output": 0.28,
        "icon": "🦈",
        "system_prompt": "You are DeepSeek V3. You are a highly intelligent advanced model."
    }
}

# --- 2. AUTHENTICATION ---
def check_password():
    def password_entered():
        if hmac.compare_digest(st.session_state["password"], st.secrets["APP_PASSWORD"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.title("🔒 Login Required")
    st.text_input("Password:", type="password", on_change=password_entered, key="password")
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("😕 Password incorrect")
    return False

if not check_password():
    st.stop()

st.title("🤖 Multi-Model Arena")

# --- 3. HELPER FUNCTIONS ---

def get_client(model_conf, user_provided_key=None):
    """
    Tries to initialize a client. 
    Priority: 
    1. User provided key (from Sidebar input)
    2. Streamlit Secrets (server-side config)
    """
    key_name = model_conf["api_key_name"]
    api_key = None

    # Check for user key first, then secret
    if user_provided_key:
        api_key = user_provided_key
    elif key_name in st.secrets:
        api_key = st.secrets[key_name]
    
    # If neither exists, return error
    if not api_key:
        return None, f"Missing Key: {key_name}"

    # Configure Google
    if model_conf["provider"] == "google":
        try:
            genai.configure(api_key=api_key)
            return genai, None
        except Exception as e:
            return None, f"Google Client Error: {str(e)}"
    
    # Configure OpenAI / Grok / DeepSeek
    else:
        try:
            base_url = model_conf.get("base_url") 
            client = OpenAI(api_key=api_key, base_url=base_url)
            return client, None
        except Exception as e:
            return None, f"Client Error: {str(e)}"

def calculate_cost(model_name, input_tokens, output_tokens):
    if model_name not in AVAILABLE_MODELS:
        return 0.0
    info = AVAILABLE_MODELS[model_name]
    in_cost = (input_tokens / 1_000_000) * info["price_input"]
    out_cost = (output_tokens / 1_000_000) * info["price_output"]
    return in_cost + out_cost

# --- 4. SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_cost" not in st.session_state:
    st.session_state.session_cost = 0.0
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0

# --- 5. AGENT CLASS ---
class Agent:
    def __init__(self, display_name, config, user_key=None):
        self.name = display_name
        self.config = config
        self.model_id = config["api_id"]
        self.provider = config["provider"]
        self.system_prompt = config["system_prompt"]
        self.avatar = config["icon"]
        
        # Initialize client immediately to check for errors
        # Pass the optional user_key to the helper function
        self.client, self.error = get_client(config, user_key)

    def generate_response(self, conversation_history):
        # If client failed to load (e.g. missing key), return error as the message
        if self.error:
            return f"⚠️ {self.error}", 0, 0

        if self.provider == "google":
            return self._call_google(conversation_history)
        else:
            return self._call_openai_compatible(conversation_history)

    def _call_openai_compatible(self, history):
        messages = [{"role": "system", "content": self.system_prompt}]
        for msg in history:
            role = "user" if msg["role"] == "user" else "assistant"
            content = f"{msg['name']}: {msg['content']}"
            messages.append({"role": role, "content": content})

        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=0.7
            )
            content = response.choices[0].message.content
            return content, response.usage.prompt_tokens, response.usage.completion_tokens
        except Exception as e:
            return f"API Error: {str(e)}", 0, 0

    def _call_google(self, history):
        try:
            model = genai.GenerativeModel(
                self.model_id,
                system_instruction=self.system_prompt
            )
            google_history = []
            for msg in history:
                role = "user" if msg["role"] == "user" else "model"
                google_history.append({"role": role, "parts": [f"{msg['name']}: {msg['content']}"]})

            response = model.generate_content(google_history)
            if not response.text:
                return "Error: Empty response.", 0, 0
                
            usage = response.usage_metadata
            return response.text, usage.prompt_token_count, usage.candidates_token_count
        except Exception as e:
            return f"Google Error: {str(e)}", 0, 0

# --- 6. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    selected_models = st.multiselect(
        "Select Participants:",
        options=list(AVAILABLE_MODELS.keys()),
        default=["GPT-4o", "Grok-3"] 
    )

    # --- DYNAMIC KEY INPUT ---
    # Container to store keys entered by the user
    user_api_keys = {}
    
    # Check if the selected models have keys in secrets. If not, ask user.
    missing_secrets = []
    for model_name in selected_models:
        key_name = AVAILABLE_MODELS[model_name]["api_key_name"]
        if key_name not in st.secrets:
            missing_secrets.append(model_name)

    if missing_secrets:
        st.divider()
        st.caption("🔑 Enter API Keys")
        st.info("Some selected models need API keys not found in system secrets.")
        
        for model_name in missing_secrets:
            key_name = AVAILABLE_MODELS[model_name]["api_key_name"]
            user_input = st.text_input(
                f"{model_name} Key", 
                type="password",
                help=f"Enter {key_name} for {model_name}"
            )
            if user_input:
                user_api_keys[model_name] = user_input
    
    # --- HELPER TEXT ---
    with st.expander("ℹ️ Where do I get keys?"):
        st.markdown("""
        If you don't have keys, sign up here:
        *   **OpenAI:** [platform.openai.com](https://platform.openai.com/api-keys)
        *   **Google:** [aistudio.google.com](https://aistudio.google.com/app/apikey)
        *   **DeepSeek:** [platform.deepseek.com](https://platform.deepseek.com)
        *   **xAI (Grok):** [console.x.ai](https://console.x.ai)
        """)

    st.divider()
    st.header("📊 Session Stats")
    st.metric("Est. Cost", f"${st.session_state.session_cost:.5f}")
    st.metric("Total Tokens", f"{st.session_state.total_tokens:,}")
    
    st.divider()
    col1, col2 = st.columns(2)
    if col1.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.session_cost = 0.0
        st.session_state.total_tokens = 0
        st.rerun()
    if col2.button("Logout"):
        st.session_state["password_correct"] = False
        st.rerun()

# --- 7. MAIN APP LOOP ---

# Instantiate active agents
active_agents = []
for name in selected_models:
    # Get user key if it exists in the dictionary, otherwise None
    user_key = user_api_keys.get(name)
    # Pass the user_key to the Agent (which will prioritize it over secrets)
    active_agents.append(Agent(name, AVAILABLE_MODELS[name], user_key))

# Render History
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message.get("avatar")):
        st.markdown(f"**{message['name']}**: {message['content']}")

# Input
if user_input := st.chat_input("Start the conversation..."):
    
    if not active_agents:
        st.error("Please select at least one AI model.")
        st.stop()

    # User
    st.session_state.messages.append({"role": "user", "name": "User", "content": user_input, "avatar": "👤"})
    with st.chat_message("user", avatar="👤"):
        st.markdown(f"**User**: {user_input}")

    # AI Response Loop
    for agent in active_agents:
        with st.chat_message("assistant", avatar=agent.avatar):
            with st.spinner(f"{agent.name} is thinking..."):
                
                content, in_tok, out_tok = agent.generate_response(st.session_state.messages)
                
                st.markdown(f"**{agent.name}**: {content}")
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "name": agent.name,
                    "content": content,
                    "avatar": agent.avatar
                })
                
                cost = calculate_cost(agent.name, in_tok, out_tok)
                st.session_state.session_cost += cost
                st.session_state.total_tokens += (in_tok + out_tok)
    
    st.rerun()
