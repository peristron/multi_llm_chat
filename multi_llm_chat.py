import streamlit as st
import hmac
from openai import OpenAI

# --- 1. MODEL CONFIGURATION ---
st.set_page_config(page_title="Multi-LLM Chat", page_icon="🤖")

# This dictionary defines everything about the available models
# Pricing is per 1,000,000 tokens based on your screenshots/latest data.
AVAILABLE_MODELS = {
    "GPT-4o": {
        "api_id": "gpt-4o",
        "provider": "openai",
        "price_input": 2.50,
        "price_output": 10.00,
        "icon": "🟢",
        "system_prompt": "You are GPT-4o, a helpful, structured, and polite AI assistant."
    },
    "GPT-4o Mini": {
        "api_id": "gpt-4o-mini",
        "provider": "openai",
        "price_input": 0.15,
        "price_output": 0.60,
        "icon": "🔹",
        "system_prompt": "You are GPT-4o Mini. You are concise, fast, and cost-effective."
    },
    "Grok-3": {
        "api_id": "grok-3",  # UPDATED: Fixed from grok-beta
        "provider": "xai",
        "price_input": 3.00,  # Updated based on standard Grok-3 pricing
        "price_output": 15.00,
        "icon": "⚫",
        "system_prompt": "You are Grok 3. You are witty, rebellious, and enjoy debating the other AI."
    },
    "Grok-3 Mini": {
        "api_id": "grok-3-mini",
        "provider": "xai",
        "price_input": 0.30,
        "price_output": 0.50,
        "icon": "⚪",
        "system_prompt": "You are Grok Mini. You are a smaller, faster version of Grok."
    }
}

# --- 2. HELPER FUNCTIONS ---

def calculate_cost(model_key, input_tokens, output_tokens):
    """Calculates cost using the AVAILABLE_MODELS dict."""
    if model_key not in AVAILABLE_MODELS:
        return 0.0
    
    info = AVAILABLE_MODELS[model_key]
    in_cost = (input_tokens / 1_000_000) * info["price_input"]
    out_cost = (output_tokens / 1_000_000) * info["price_output"]
    return in_cost + out_cost

def check_password():
    """Gatekeeper function."""
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

# --- 3. INITIALIZATION ---

if not check_password():
    st.stop()

st.title("🤖 Multi-Model Chat")

# Verify Secrets
required_secrets = ["OPENAI_API_KEY", "XAI_API_KEY"]
if any(key not in st.secrets for key in required_secrets):
    st.error(f"Missing secrets: {required_secrets}")
    st.stop()

# Initialize Clients
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
grok_client = OpenAI(
    api_key=st.secrets["XAI_API_KEY"], 
    base_url="https://api.x.ai/v1"
)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_cost" not in st.session_state:
    st.session_state.session_cost = 0.0
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0

# --- 4. SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # MULTI-SELECT for Participants
    # This allows you to choose who enters the ring
    selected_models = st.multiselect(
        "Select Participants:",
        options=list(AVAILABLE_MODELS.keys()),
        default=["GPT-4o", "Grok-3"] # Default selection
    )
    
    st.divider()
    st.header("📊 Session Stats")
    st.metric("Est. Cost", f"${st.session_state.session_cost:.5f}")
    st.metric("Total Tokens", f"{st.session_state.total_tokens:,}")
    
    st.divider()
    if st.button("Clear Chat & Reset Cost"):
        st.session_state.messages = []
        st.session_state.session_cost = 0.0
        st.session_state.total_tokens = 0
        st.rerun()
    if st.button("Logout"):
        st.session_state["password_correct"] = False
        st.rerun()

# --- 5. AGENT CLASS & INSTANTIATION ---

class Agent:
    def __init__(self, display_name, client, model_config):
        self.name = display_name
        self.client = client
        self.model = model_config["api_id"]
        self.system_prompt = model_config["system_prompt"]
        self.avatar = model_config["icon"]

    def generate_response(self, conversation_history):
        messages = [{"role": "system", "content": self.system_prompt}]
        for msg in conversation_history:
            role = "user" if msg["role"] == "user" else "assistant"
            content = f"{msg['name']}: {msg['content']}"
            messages.append({"role": role, "content": content})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7
            )
            content = response.choices[0].message.content
            usage = response.usage
            return content, usage
        except Exception as e:
            return f"Error: {str(e)}", None

# Dynamically create the list of active agents based on Sidebar Selection
active_agents = []
for model_name in selected_models:
    config = AVAILABLE_MODELS[model_name]
    
    # Select the correct client based on provider
    if config["provider"] == "openai":
        client_to_use = openai_client
    else:
        client_to_use = grok_client
        
    active_agents.append(Agent(model_name, client_to_use, config))

# --- 6. CHAT INTERFACE ---

# Display History
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message.get("avatar")):
        st.markdown(f"**{message['name']}**: {message['content']}")

# Input
if user_input := st.chat_input("Start the conversation..."):
    
    if not active_agents:
        st.error("Please select at least one participant in the sidebar!")
        st.stop()

    # User speaks
    st.session_state.messages.append({"role": "user", "name": "User", "content": user_input, "avatar": "👤"})
    with st.chat_message("user", avatar="👤"):
        st.markdown(f"**User**: {user_input}")

    # Agents speak
    for agent in active_agents:
        with st.chat_message("assistant", avatar=agent.avatar):
            with st.spinner(f"{agent.name} is thinking..."):
                response_text, usage = agent.generate_response(st.session_state.messages)
                
                st.markdown(f"**{agent.name}**: {response_text}")
                
                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "name": agent.name,
                    "content": response_text,
                    "avatar": agent.avatar
                })
                
                # Calculate Cost
                if usage:
                    cost = calculate_cost(agent.name, usage.prompt_tokens, usage.completion_tokens)
                    st.session_state.session_cost += cost
                    st.session_state.total_tokens += usage.total_tokens
    
    st.rerun()
