import streamlit as st
import requests
import json
import time

# ----------------------------
# Streamlit Page Config
# ----------------------------
st.set_page_config(
    page_title="📰 AI News Analyst",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Custom CSS for a modern look
# ----------------------------
st.markdown("""
<style>
    /* Main app styling */
    .stApp {
        background-color: #f0f2f6;
    }
    
    /* Title */
    .st-emotion-cache-10trblm {
        color: #1a1a1a;
    }

    /* Sidebar */
    .st-emotion-cache-16txtl3 {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    
    /* Text Area */
    .stTextArea textarea {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
    }
    
    /* Button */
    .stButton button {
        background-color: #0068c9;
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: bold;
    }
    .stButton button:hover {
        background-color: #0056b3;
        color: white;
    }
    
    /* Result tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
    }

    /* Result Box */
    .result-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        min-height: 200px;
    }
    
    /* Source/Citation styling */
    .source-link {
        color: #0068c9;
        text-decoration: none;
    }
    .source-title {
        font-weight: bold;
        color: #333;
    }
    .source-item {
        margin-bottom: 10px;
        padding: 10px;
        background-color: #f9f9f9;
        border-radius: 5px;
    }

</style>
""", unsafe_allow_html=True)

# ----------------------------
# AI System Prompt
# ----------------------------
SYSTEM_PROMPT = """
You are an expert, meticulous, and unbiased AI News Analyst. Your sole purpose is to analyze a news article provided by the user and determine its authenticity and bias.

You MUST use the provided Google Search tool. Your analysis is worthless without real-time, external verification.

**Analysis Steps:**
1.  **Identify Core Claims:** What are the central factual claims of the article (who, what, when, where, why)?
2.  **Search & Corroborate:** Use Google Search to find these core claims on high-authority, independent news domains (e.g., Reuters, AP, BBC, NYT, WSJ).
3.  **Search for Debunks:** Actively search for the claims on fact-checking websites (e.g., Snopes, PolitiFact, FactCheck.org, AP Fact Check).
4.  **Analyze Tone & Bias:** Analyze the provided text for emotionally manipulative language, loaded words, or one-sided framing.
5.  **Formulate Verdict:** Based on your findings, provide a clear verdict.

**Response Format:**
You MUST provide your response in this exact Markdown format:

**Verdict:** [Choose one: ✅ **Verified**, ⚠️ **Unproven**, 🟠 **Misleading/Out of Context**, ❌ **Likely False**]

**Analysis:**
* **[Bullet point 1 - Your main finding]**
* **[Bullet point 2 - Details on corroboration or lack thereof]**
* **[Bullet point 3 - Analysis of bias, tone, or other red flags]**
* **[Bullet point 4 - Any other relevant findings]**
"""

# ----------------------------
# Gemini API Call Function
# ----------------------------
def get_gemini_analysis(api_key, article_text):
    if not api_key:
        st.error("API key is missing. Please enter your API key in the sidebar.")
        return None, None

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    
    payload = {
        "contents": [{"parts": [{"text": article_text}]}],
        "tools": [{"google_search": {}}],
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
    }

    max_retries = 3
    backoff_time = 2  # Start with 2 seconds

    for attempt in range(max_retries):
        try:
            response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=120)
            
            if response.status_code == 429:
                # Rate limited
                st.warning(f"Rate limited. Retrying in {backoff_time} seconds...")
                time.sleep(backoff_time)
                backoff_time *= 2  # Exponential backoff
                continue

            response.raise_for_status()
            response_json = response.json()
            
            # --- Extract Text ---
            candidate = response_json.get("candidates", [])[0]
            if "content" not in candidate:
                return "Error: The model's response was blocked. The input may contain sensitive content or the model's safety settings were triggered.", None
                
            text_response = candidate.get("content", {}).get("parts", [])[0].get("text", "")
            
            # --- Extract Sources ---
            grounding_metadata = candidate.get("groundingMetadata", {})
            sources = []
            if "groundingAttributions" in grounding_metadata:
                for attribution in grounding_metadata["groundingAttributions"]:
                    if "web" in attribution:
                        sources.append({
                            "title": attribution["web"].get("title", "No Title Found"),
                            "uri": attribution["web"].get("uri", "#")
                        })
            
            if not text_response:
                text_response = "Error: Received an empty response. The model may have had an issue."

            return text_response, sources

        except requests.exceptions.HTTPError as http_err:
            return f"Error: HTTP Error. Status: {http_err.response.status_code}. Details: {http_err.response.text}", None
        except requests.exceptions.RequestException as req_err:
            return f"Error: Network connection failed: {req_err}", None
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            st.error(f"Error parsing AI response: {e} - Response: {response.text}")
            return "Error: Could not parse the AI's response.", None

    return "Error: The request timed out after multiple retries. The server might be busy.", None


# ----------------------------
# Streamlit UI
# ----------------------------

# --- Sidebar ---
with st.sidebar:
    st.header("🔑 API Configuration")
    st.markdown("You need a Google Gemini API key to use this app.")
    api_key = st.text_input("Enter your Google Gemini API Key:", type="password", help="Get your free key from Google AI Studio")
    if not api_key:
        st.warning("Please enter your API key to activate the analysis.")
    st.markdown("[Get your API key from Google AI Studio](https://aistudio.google.com/app/apikey)")
    st.markdown("---")
    st.subheader("Why this is better:")
    st.markdown("""
    * **Old Way (Static):** Used an old `.csv` file. It was "blind" to new events and could only guess based on writing style.
    * **New Way (Dynamic):** Uses a live AI with **Google Search**. It can check facts against *today's* news and provide real-time evidence.
    """)

# --- Main Page ---
st.title("📰 AI News Analyst")
st.markdown("Beyond 'Fake' or 'Real': Get a Real-Time Analysis with Verifiable Sources.")

st.markdown("### 1. Paste Your Article")
user_input = st.text_area("Paste the full news article text here:", height=250, 
                          placeholder="Example: 'A new study from Harvard University claims that coffee can make you fly...'")

if st.button("🚀 Analyze Article Now"):
    if not api_key:
        st.error("Please enter your Gemini API key in the sidebar to proceed.")
    elif not user_input.strip() or len(user_input.strip()) < 50:
        st.warning("⚠️ Please paste the full text of the article (at least 50 characters) to get a proper analysis.")
    else:
        with st.spinner("🧠 AI is performing live searches and analyzing the article... This may take a moment."):
            analysis, sources = get_gemini_analysis(api_key, user_input)
            
            st.markdown("### 2. Analysis Results")
            
            tab1, tab2, tab3 = st.tabs(["🔍 AI Analysis & Verdict", "📚 Sources Found", "⚠️ How This Works (Disclaimer)"])
            
            with tab1:
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                if analysis:
                    st.markdown(analysis)
                else:
                    st.error("Could not retrieve analysis. Please try again.")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with tab2:
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                if sources:
                    st.markdown(f"**The AI found {len(sources)} sources to corroborate its analysis:**")
                    st.markdown("---")
                    for i, source in enumerate(sources):
                        st.markdown(f"""
                        <div class="source-item">
                            <span class="source-title">{i+1}. {source['title']}</span><br>
                            <a href="{source['uri']}" target="_blank" class="source-link">{source['uri']}</a>
                        </div>
                        """, unsafe_allow_html=True)
                elif "Error" not in (analysis or ""):
                    st.info("The AI completed its analysis without needing to cite specific external web sources, or no external sources were found for the claims.")
                else:
                    st.info("No sources to display.")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with tab3:
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.warning("**Important Limitation: No AI is 100% Accurate**")
                st.markdown("""
                This tool is an *analyst*, not an *oracle*. It is designed to assist a human (you) by doing rapid research and analysis.
                
                * **Always** read the AI's analysis and check the sources yourself.
                * The verdict is based on information available on the public internet *at this moment*.
                * "Fake news" is designed to be deceptive. A well-written fake article may temporarily trick both people and AI.
                
                **For your presentation:** Explain that the *real power* of this tool is the **transparent analysis** and **verifiable sources**, not just a simple "fake" or "real" label.
                """)
                st.markdown('</div>', unsafe_allow_html=True)
