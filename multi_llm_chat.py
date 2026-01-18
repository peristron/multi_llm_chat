import streamlit as st
import hmac
from openai import OpenAI

# --- 1. CONFIGURATION & PRICING ---
st.set_page_config(page_title="Multi-LLM Chat", page_icon="🤖")

# Pricing Dictionary (Per 1,000,000 tokens) based on your screenshots
# We map the Model ID string to its pricing structure.
MODEL_PRICING = {
    "gpt-4o": {
        "input": 2.50,   # $2.50 per 1M input tokens
        "output": 10.00  # $10.00 per 1M output tokens
    },
    # Using Grok-2 pricing from screenshot as the reference for Grok
    "grok-beta": {
        "input": 2.00,   # $2.00 per 1M
        "output": 10.00  # $10.00 per 1M
    },
    "grok-2-1212": {
        "input": 2.00,
        "output": 10.00
    }
}

def calculate_cost(model_name, input_tokens, output_tokens):
    """Calculates cost for a single transaction."""
    if model_name not in MODEL_PRICING:
        return 0.0
    
    prices = MODEL_PRICING[model_name]
    in_cost = (input_tokens / 1_000_000) * prices["input"]
    out_cost = (output_tokens / 1_000_000) * prices["output"]
    
    return in_cost + out_cost

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

# --- 3. SETUP & CLIENTS ---
st.title("🤖 Multi-Model Chat")

# Validate Secrets
required_secrets = ["OPENAI_API_KEY", "XAI_API_KEY"]
if any(key not in st.secrets for key in required_secrets):
    st.error(f"Missing secrets. Ensure {required_secrets} are set.")
    st.stop()

# Initialize Clients
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
grok_client = OpenAI(
    api_key=st.secrets["XAI_API_KEY"], 
    base_url="https://api.x.ai/v1"
)

# Initialize Session State for Cost & Messages
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_cost" not in st.session_state:
    st.session_state.session_cost = 0.0
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0

# --- 4. AGENT CLASS ---
class Agent:
    def __init__(self, name, client, model, system_prompt, avatar):
        self.name = name
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.avatar = avatar

    def generate_response(self, conversation_history):
        messages = [{"role": "system", "content": self.system_prompt}]
        for msg in conversation_history:
            role = "user" if msg["role"] == "user" else "assistant"
            content = f"{msg['name']}: {msg['content']}"
            messages.append({"role": role, "content": content})

        try:
            # We need the full response object to get usage stats
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            usage = response.usage # Contains prompt_tokens and completion_tokens
            
            return content, usage
            
        except Exception as e:
            return f"Error: {str(e)}", None

# Define Agents
# Note: Ensure these model names match the keys in MODEL_PRICING dictionary
agents = [
    Agent(
        name="GPT-4o",
        client=openai_client,
        model="gpt-4o", 
        system_prompt="You are a helpful, structured AI assistant from OpenAI.",
        avatar="🟢"
    ),
    Agent(
        name="Grok",
        client=grok_client,
        model="grok-beta", 
        system_prompt="You are Grok. You are witty, rebellious, and enjoy debating.",
        avatar="⚫"
    )
]

# --- 5. SIDEBAR (COST TRACKER) ---
with st.sidebar:
    st.header("📊 Session Stats")
    
    # Format cost to 5 decimal places to catch micro-costs
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

# --- 6. CHAT INTERFACE ---

# Render History
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message.get("avatar")):
        st.markdown(f"**{message['name']}**: {message['content']}")

# Input Handler
if user_input := st.chat_input("Start the conversation..."):
    
    # A. User speaks
    st.session_state.messages.append({"role": "user", "name": "User", "content": user_input, "avatar": "👤"})
    with st.chat_message("user", avatar="👤"):
        st.markdown(f"**User**: {user_input}")

    # B. Agents respond
    for agent in agents:
        with st.chat_message("assistant", avatar=agent.avatar):
            with st.spinner(f"{agent.name} is thinking..."):
                
                # Generate Response & Get Usage
                response_text, usage = agent.generate_response(st.session_state.messages)
                
                # Display text
                st.markdown(f"**{agent.name}**: {response_text}")
                
                # Update History
                st.session_state.messages.append({
                    "role": "assistant",
                    "name": agent.name,
                    "content": response_text,
                    "avatar": agent.avatar
                })
                
                # Update Cost (If usage data exists)
                if usage:
                    cost = calculate_cost(agent.model, usage.prompt_tokens, usage.completion_tokens)
                    st.session_state.session_cost += cost
                    st.session_state.total_tokens += usage.total_tokens
                    
    # Force a rerun to update the sidebar metrics immediately after the loop
    st.rerun()
