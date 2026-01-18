import streamlit as st
import hmac
from openai import OpenAI

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Multi-LLM Chat",
    page_icon="🤖",
    layout="centered"
)

# --- AUTHENTICATION LOGIC ---
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["APP_PASSWORD"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't keep password in state
        else:
            st.session_state["password_correct"] = False

    # Return True if the user has already validated
    if st.session_state.get("password_correct", False):
        return True

    # Show input if password hasn't been entered yet
    st.title("🔒 Login Required")
    st.text_input(
        "Please enter the application password:", 
        type="password", 
        on_change=password_entered, 
        key="password"
    )
    
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("😕 Password incorrect")

    return False

# Stop execution if password is not correct
if not check_password():
    st.stop()

# --- MAIN APP START (Only runs if authenticated) ---

st.title("🤖 Multi-Model Chat")
st.caption("Authenticated Session: You + GPT-4o + Grok")

# --- 1. CREDENTIAL CHECK ---
# Ensure all required keys exist in secrets
required_secrets = ["OPENAI_API_KEY", "GROK_API_KEY", "APP_PASSWORD"]
missing_secrets = [key for key in required_secrets if key not in st.secrets]

if missing_secrets:
    st.error(f"Missing secrets in configuration: {', '.join(missing_secrets)}")
    st.stop()

# Initialize Clients
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Grok (xAI) Client
grok_client = OpenAI(
    api_key=st.secrets["GROK_API_KEY"], 
    base_url="https://api.x.ai/v1"
)

# --- 2. AGENT CLASS DEFINITION ---
class Agent:
    def __init__(self, name, client, model, system_prompt, avatar):
        self.name = name
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.avatar = avatar

    def generate_response(self, conversation_history):
        # System prompt first
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Flatten history for the API
        for msg in conversation_history:
            role = "user" if msg["role"] == "user" else "assistant"
            # We explicitly put the Name in the content so the models know who is talking
            content = f"{msg['name']}: {msg['content']}"
            messages.append({"role": role, "content": content})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"

# Define Agents
agents = [
    Agent(
        name="GPT-4o",
        client=openai_client,
        model="gpt-4o", 
        system_prompt="You are a helpful, structured AI assistant. You answer questions concisely.",
        avatar="🟢"
    ),
    Agent(
        name="Grok",
        client=grok_client,
        model="grok-beta", 
        system_prompt="You are Grok. You are witty, rebellious, and enjoy debating. You often comment on the previous AI's response.",
        avatar="⚫"
    )
]

# --- 3. CHAT HISTORY STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 4. RENDER UI ---
# Render previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message.get("avatar")):
        st.markdown(f"**{message['name']}**: {message['content']}")

# Input area
if user_input := st.chat_input("Start the conversation..."):
    
    # 1. Add User Message
    st.session_state.messages.append({
        "role": "user",
        "name": "User",
        "content": user_input,
        "avatar": "👤"
    })
    with st.chat_message("user", avatar="👤"):
        st.markdown(f"**User**: {user_input}")

    # 2. Iterate through Agents
    for agent in agents:
        with st.chat_message("assistant", avatar=agent.avatar):
            with st.spinner(f"{agent.name} is thinking..."):
                # Pass full history (User + Previous Agents) to current Agent
                response_text = agent.generate_response(st.session_state.messages)
                
                st.markdown(f"**{agent.name}**: {response_text}")
                
                # Append to history for next agent/loop
                st.session_state.messages.append({
                    "role": "assistant",
                    "name": agent.name,
                    "content": response_text,
                    "avatar": agent.avatar
                })

# --- 5. SIDEBAR ---
with st.sidebar:
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    if st.button("Logout"):
        st.session_state["password_correct"] = False
        st.rerun()
