import streamlit as st
import hmac
import re  # added for string cleaning
from openai import OpenAI
import google.generativeai as genai

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Multi-LLM Chat", page_icon="🤖", layout="wide")

# defining the models
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

# --- 2. auth
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

# --- app instructions-
with st.expander("📚 How to use this app"):
    st.markdown("""
    *   **Select Participants:** Choose which AI models to chat with in the Sidebar.
    *   **Directed Chat:** By default, all selected models respond. Type **`@grok`**, **`@gpt`**, or **`@deepseek`** to target specific models.
    *   **Persistent Memory:** The conversation history is sent with every message. Instructions like *"be concise"* or *"answer in JSON"* will be remembered for the whole session.
    *   **Smart Fallback:** If a specific Google model (e.g. Pro) is unavailable, the app automatically switches to the best available alternative (e.g. Flash).
    """)

# --- 3. helper functions

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
    
    # configuring openAI / grok / deepSeek
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
    def __init__(self, display_name, config, user_key=None, concise_mode=True):
        self.name = display_name
        self.config = config
        self.model_id = config["api_id"]
        self.provider = config["provider"]
        self.avatar = config["icon"]
        
        # --- system prompt logic
        base_prompt = config["system_prompt"]
        if concise_mode:
            base_prompt += " Your default behavior is to be brief, concise, and direct. Only provide long or detailed explanations if the user explicitly asks for them."
            
        self.system_prompt = base_prompt
        
        # initializes client immediately to check for errors
        self.client, self.error = get_client(config, user_key)

    def _clean_response(self, text):
        """
        Removes the model's own name from the start of the response.
        Example: "GPT-4o: Hello" -> "Hello"
        """
        if not text:
            return ""
            
        # Rregex to match "Name:" or "Name " or "Name-" at start of string, case insensitive
        #escape the name to handle dots/spaces in model names
        pattern = r"^" + re.escape(self.name) + r"[:\-\s]+"
        cleaned_text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
        return cleaned_text

    def generate_response(self, conversation_history):
        # if client failed to load (e.g. missing key), return error as the message
        if self.error:
            return f"⚠️ {self.error}", 0, 0

        if self.provider == "google":
            content, in_tok, out_tok = self._call_google(conversation_history)
        else:
            content, in_tok, out_tok = self._call_openai_compatible(conversation_history)
            
        # cleans the "Name Echo" before returning
        return self._clean_response(content), in_tok, out_tok

    def _call_openai_compatible(self, history):
        messages = [{"role": "system", "content": self.system_prompt}]
        for msg in history:
            role = "user" if msg["role"] == "user" else "assistant"
            content = f"{msg['name']}: {msg['content']}"
            messages.append({"role": role, "content": content})

        try:
            # STOP SEQUENCES: prevent deepSeek/grok from roleplaying other bots
            stop_sequences = ["User:", "User", "\nUser"]
            for model_name in AVAILABLE_MODELS.keys():
                stop_sequences.append(f"{model_name}:")
                stop_sequences.append(f"\n{model_name}:")

            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=0.7,
                stop=stop_sequences[:4] # openAI limits stop sequences (usually 4)
                # note: DeepSeek usually supports more, but sticking to 4 key ones or just "User:" 
                # is often enough if the system prompt is strong
                # tries just catching the most likely ones to break the loop
            )
            content = response.choices[0].message.content
            return content, response.usage.prompt_tokens, response.usage.completion_tokens
        except Exception as e:
            return f"API Error: {str(e)}", 0, 0

    def _call_google(self, history):
        def execute_gemini(model_name, chat_history):
            model = genai.GenerativeModel(
                model_name,
                system_instruction=self.system_prompt
            )
            google_history = []
            for msg in chat_history:
                role = "user" if msg["role"] == "user" else "model"
                google_history.append({"role": role, "parts": [f"{msg['name']}: {msg['content']}"]})

            # 
            # ONLY stop on "User:". 
            # DO NOT stop on self.name (e.g. "Gemini:") because if the model 
            # starts with its own name, the API cuts it off at token 0 and crashes
            gen_config = genai.types.GenerationConfig(
                stop_sequences=["User:", "User", "\nUser"]
            )

            response = model.generate_content(google_history, generation_config=gen_config)
            
            # checks for valid text parts
            if not response.candidates or not response.candidates[0].content.parts:
                # if the model blocked itself or returned empty, return a fallback message
                # prevents the app from crashing on safety/empty blocks
                return "...", 0, 0
            
            usage = response.usage_metadata
            return response.text, usage.prompt_token_count, usage.candidates_token_count

        try:
            return execute_gemini(self.model_id, history)
        except Exception as e:
            error_str = str(e).lower()
            if "404" in error_str or "not found" in error_str:
                try:
                    available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                    clean_available = [m.replace("models/", "") for m in available_models]

                    fallback_model = None
                    if "gemini-1.5-pro" in clean_available: fallback_model = "gemini-1.5-pro"
                    elif "gemini-1.5-flash" in clean_available: fallback_model = "gemini-1.5-flash"
                    elif "gemini-pro" in clean_available: fallback_model = "gemini-pro"
                    elif len(available_models) > 0: fallback_model = available_models[0].replace("models/", "")

                    if fallback_model:
                        text, in_tok, out_tok = execute_gemini(fallback_model, history)
                        text += f"\n\n*(Note: {self.model_id} was unavailable. Auto-switched to {fallback_model})*"
                        return text, in_tok, out_tok
                    else:
                        return f"Google Error: Model not found and no valid fallbacks available.", 0, 0
                except Exception as fallback_error:
                    return f"Google Error: {str(e)} | Fallback failed: {str(fallback_error)}", 0, 0
            return f"Google Error: {str(e)}", 0, 0

# --- 6. sidebar-
with st.sidebar:
    st.header("⚙️ Configuration")
    
    selected_models = st.multiselect(
        "Select Participants (add as needed):",
        options=list(AVAILABLE_MODELS.keys()),
        default=["GPT-4o", "Grok-3"] 
    )

    st.caption("Settings")
    concise_mode = st.checkbox(
        "Concise Responses", 
        value=True, 
        help="If checked, instructs all models to keep answers brief and direct unless asked otherwise."
    )

    # --- dynamic key entry
    user_api_keys = {}
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
            user_input = st.text_input(f"{model_name} Key", type="password")
            if user_input:
                user_api_keys[model_name] = user_input
    
    # --helper text
    with st.expander("ℹ️ Where do I get keys?"):
        st.markdown("""
        If you don't have keys, sign up here (just some popular examples):
        *   **OpenAI:** [platform.openai.com](https://platform.openai.com/api-keys)
        *   **Google:** [aistudio.google.com](https://aistudio.google.com/app/apikey)
        *   **DeepSeek:** [platform.deepseek.com](https://platform.deepseek.com)
        *   **xAI (Grok):** [console.x.ai](https://console.x.ai)
        """)

    st.divider()
    st.header("📊 Session Stats/Cost Calculator")
    st.metric("Est. Cost", f"${st.session_state.session_cost:.5f}")
    st.caption("Aggregated total across all models used in the chat session. Estimates use standard pricing.")
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

# --- 7. main app loop

# instantiating active agents
active_agents = []
for name in selected_models:
    user_key = user_api_keys.get(name)
    active_agents.append(Agent(name, AVAILABLE_MODELS[name], user_key, concise_mode))

# render history
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message.get("avatar")):
        st.markdown(f"**{message['name']}**: {message['content']}")

# input
if user_input := st.chat_input("Start the conversation... (Tip: use @mention to target specific models like '@deepseek')"):
    
    if not active_agents:
        st.error("Please select at least one AI model.")
        st.stop()

    # user
    st.session_state.messages.append({"role": "user", "name": "User", "content": user_input, "avatar": "👤"})
    with st.chat_message("user", avatar="👤"):
        st.markdown(f"**User**: {user_input}")

    # --mention logic
    responders = active_agents
    mentioned_agents = []
    lower_input = user_input.lower()
    for agent in active_agents:
        name_lower = agent.name.lower()
        triggers = [f"@{name_lower}", f"@{name_lower.replace(' ', '')}", f"@{name_lower.split()[0]}", f"@{name_lower.split('-')[0]}"]
        if any(trigger in lower_input for trigger in triggers):
            mentioned_agents.append(agent)
    if mentioned_agents:
        responders = mentioned_agents

    # ai response loop
    for agent in responders:
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
