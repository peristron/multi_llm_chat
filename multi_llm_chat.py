import streamlit as st
from openai import OpenAI

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Multi-LLM Chat",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Multi-Model Chat: OpenAI & Grok")
st.caption("A 3-way conversation between you, GPT-4o, and Grok.")

# --- 1. SETUP & CREDENTIALS ---
# We check if secrets are loaded. If running locally, this looks at .streamlit/secrets.toml
# If running on Community Cloud, this looks at the App Settings -> Secrets area.

if "OPENAI_API_KEY" not in st.secrets or "GROK_API_KEY" not in st.secrets:
    st.error("API Keys are missing! Please set OPENAI_API_KEY and GROK_API_KEY in your Streamlit secrets.")
    st.stop()

# Initialize Clients
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Grok uses the OpenAI SDK but points to xAI's base URL
grok_client = OpenAI(
    api_key=st.secrets["GROK_API_KEY"], 
    base_url="https://api.x.ai/v1"
)

# --- 2. AGENT DEFINITIONS ---
class Agent:
    def __init__(self, name, client, model, system_prompt, avatar, color):
        self.name = name
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.avatar = avatar
        self.color = color

    def generate_response(self, conversation_history):
        # 1. Start with the agent's specific system instructions
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # 2. Add the conversation history
        # We format the history so the AI understands who said what
        for msg in conversation_history:
            # We map "assistant" roles to generic inputs, but in the content
            # we explicitly label who is speaking to avoid confusion.
            role = "user" if msg["role"] == "user" else "assistant"
            content = f"{msg['name']}: {msg['content']}"
            messages.append({"role": role, "content": content})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7 # Creativity level
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"

# Define our two agents
agents = [
    Agent(
        name="GPT-4o",
        client=openai_client,
        model="gpt-4o", 
        system_prompt="You are a helpful, logical, and polite AI assistant. You value structure and facts.",
        avatar="🟢",
        color="green"
    ),
    Agent(
        name="Grok",
        client=grok_client,
        model="grok-beta", # Ensure this model name matches current xAI availability
        system_prompt="You are Grok. You have a rebellious, witty personality. You like to debate and offer alternative, sometimes edgy viewpoints compared to standard AI responses.",
        avatar="⚫",
        color="grey"
    )
]

# --- 3. SESSION STATE (HISTORY) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 4. RENDER CHAT HISTORY ---
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message.get("avatar")):
        st.write(f"**{message['name']}**: {message['content']}")

# --- 5. CHAT LOGIC ---
if user_input := st.chat_input("Say something to the group..."):
    
    # A. Append User Message
    st.session_state.messages.append({
        "role": "user",
        "name": "User",
        "content": user_input,
        "avatar": "👤"
    })
    
    # Display immediately
    with st.chat_message("user", avatar="👤"):
        st.write(f"**User**: {user_input}")

    # B. Trigger Agents Sequence
    # This loop ensures every agent gets a turn to speak based on the updated history
    for agent in agents:
        with st.chat_message("assistant", avatar=agent.avatar):
            with st.spinner(f"{agent.name} is thinking..."):
                
                # Prepare history for the API (exclude system prompts of other agents)
                # We simply pass the list of previous messages stored in session state
                response_text = agent.generate_response(st.session_state.messages)
                
                # Display response
                st.write(f"**{agent.name}**: {response_text}")
                
                # Add to history so the NEXT agent sees this response
                st.session_state.messages.append({
                    "role": "assistant",
                    "name": agent.name,
                    "content": response_text,
                    "avatar": agent.avatar
                })

# --- 6. SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("Settings")
    if st.button("Reset Conversation"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.markdown("### How it works")
    st.markdown("1. You send a message.")
    st.markdown("2. **GPT-4o** responds to you.")
    st.markdown("3. **Grok** responds to *both* you and GPT-4o.")
