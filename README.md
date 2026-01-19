🤖 Multi-Model Arena

Multi-Model Arena is a Streamlit application that allows users to chat with multiple leading Large Language Models (LLMs) simultaneously. It is designed for benchmarking, comparing writing styles, and analyzing reasoning capabilities across different providers in a single, unified interface.
✨ Features

    Simultaneous Generation: Prompt once, receive responses from GPT-4o, Grok-3, Gemini 1.5, and DeepSeek V3 at the same time.
    Hybrid Authentication: secure API key management. Keys can be stored in server-side secrets or entered manually by the user via the sidebar.
    Smart Fallback (Google): Automatically detects if your Google API key lacks access to "Pro" models and seamlessly switches to "Flash" or other available models without crashing.
    Cost Estimation: Tracks real-time token usage and provides an aggregated session cost estimate based on current model pricing.
    Secure Access: Includes a lightweight password protection screen to prevent unauthorized usage.

🚀 Supported Models

    OpenAI: GPT-4o
    xAI: Grok-3
    Google: Gemini 1.5 Pro (with auto-fallback to Flash)
    DeepSeek: DeepSeek V3

🛠️ Installation

    Clone the repository:

    Bash

    git clone https://github.com/your-username/multi-model-arena.git
    cd multi-model-arena

    Install dependencies:

    Bash

    pip install -r requirements.txt

    Ensure your requirements.txt contains:

    text

    streamlit
    openai>=1.0.0
    google-generativeai

⚙️ Configuration
1. Secrets Management

The app uses Streamlit Secrets for configuration. If running locally, create a file at .streamlit/secrets.toml. If deploying to Streamlit Cloud, add these to the "Secrets" settings.

.streamlit/secrets.toml template:

toml

# Password for App Access
APP_PASSWORD = "your-app-login-password"

# API Keys (Optional - if left blank, users will be asked to enter them in the UI)
OPENAI_API_KEY = "sk-..."
XAI_API_KEY = "xai-..."
GOOGLE_API_KEY = "AIza..."
DEEPSEEK_API_KEY = "sk-..."

2. Customizing Models

You can easily add or remove models by modifying the AVAILABLE_MODELS dictionary in multi_llm_chat.py. The app supports any provider compatible with the OpenAI SDK (just add the base_url).
🖥️ Usage

Run the app locally:

Bash

streamlit run multi_llm_chat.py

    Enter the app password (defined in your secrets).
    Select which models you want to participate in the sidebar.
    If any API keys are missing from the secrets file, input fields will automatically appear in the sidebar.
    Start chatting!

💰 Notes on Cost Estimation

The "Est. Cost" metric in the sidebar is an aggregated total of all active models in the session.

    Note: To ensure safety, the app estimates costs based on the standard pricing of the selected model. If the app performs a fallback (e.g., switching from Gemini Pro to Gemini Flash), the displayed estimate may be slightly higher than the actual billed amount.

📄 License

MIT License
