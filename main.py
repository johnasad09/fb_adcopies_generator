"""
Product/Service Extractor & Facebook Ad Copy Generator
====================================================

A Streamlit application that:
1. Extracts products/services from any website URL
2. Generates Facebook ad copies for selected products using AI
3. Supports multiple LLM providers (Google Gemini, OpenAI)

Dependencies: streamlit, langchain, langchain-google-genai, langchain-openai, langchain-community
"""

import streamlit as st
import json
import re
from urllib.parse import urlparse
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader

# =============================================
# CONFIGURATION & INITIALIZATION
# =============================================

# Initialize session state
if 'products' not in st.session_state:
    st.session_state.products = []
if 'processed_url' not in st.session_state:
    st.session_state.processed_url = ""
if 'ad_copies' not in st.session_state:
    st.session_state.ad_copies = ""
if 'selected_llm' not in st.session_state:
    st.session_state.selected_llm = "Google Gemini"
if 'session_api_key' not in st.session_state:
    st.session_state.session_api_key = ""


# =============================================
# LLM CONFIGURATION FUNCTIONS
# =============================================

def get_llm_instance():
    """
    Get the appropriate LLM instance based on user selection.

    Returns:
        LLM instance based on selection and available API key

    Raises:
        ValueError: If no API key is provided for selected model
    """
    if not st.session_state.session_api_key.strip():
        raise ValueError("API key is required for the selected model")

    if st.session_state.selected_llm == "Google Gemini":
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            api_key=st.session_state.session_api_key
        )
    elif st.session_state.selected_llm == "OpenAI GPT":
        return ChatOpenAI(
            model="gpt-3.5-turbo",
            api_key=st.session_state.session_api_key,
            temperature=0.7
        )
    else:
        raise ValueError(f"Unsupported LLM: {st.session_state.selected_llm}")


def display_llm_selector():
    """
    Display LLM selection interface in sidebar.
    """
    st.sidebar.header("🤖 AI Model Selection")

    llm_options = ["Google Gemini", "OpenAI GPT"]
    selected = st.sidebar.selectbox(
        "Choose AI Model:",
        options=llm_options,
        index=llm_options.index(st.session_state.selected_llm)
    )

    # Update session state if selection changed
    if selected != st.session_state.selected_llm:
        st.session_state.selected_llm = selected
        st.session_state.session_api_key = ""  # Reset API key when model changes
        st.rerun()

    # Show API key input
    provider_name = "Google" if st.session_state.selected_llm == "Google Gemini" else "OpenAI"
    api_key_input = st.sidebar.text_input(
        f"Enter {provider_name} API Key:",
        type="password",
        value=st.session_state.session_api_key,
        help="This key will only be stored for the current session"
    )

    # Update session API key
    if api_key_input != st.session_state.session_api_key:
        st.session_state.session_api_key = api_key_input

    # Show status
    if st.session_state.session_api_key:
        st.sidebar.success("✅ API Key provided")
    else:
        st.sidebar.error("❌ API Key required")


    st.sidebar.markdown("---")


def is_llm_ready():
    """
    Check if the selected LLM is ready to use.

    Returns:
        bool: True if LLM is ready, False otherwise
    """
    return bool(st.session_state.session_api_key.strip())


def get_current_model_info():
    """
    Get information about the currently active model.

    Returns:
        str: Model information string
    """
    if st.session_state.selected_llm == "Google Gemini" and st.session_state.session_api_key:
        return "🔷 **Active Model:** Google Gemini Flash 1.5"
    elif st.session_state.selected_llm == "OpenAI GPT" and st.session_state.session_api_key:
        return "🔶 **Active Model:** OpenAI GPT-3.5-Turbo"
    else:
        return f"⚠️ **Configuration Required:** Please provide API key for {st.session_state.selected_llm}"


# =============================================
# UTILITY FUNCTIONS
# =============================================

def normalize_url(url: str) -> str:
    """
    Normalize URL by adding https:// if no scheme is provided.

    Args:
        url (str): Raw URL input

    Returns:
        str: Normalized URL with https:// scheme
    """
    url = url.strip()
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    return url


def is_valid_url(url: str) -> bool:
    """
    Validate if the provided string is a valid URL.

    Args:
        url (str): URL to validate

    Returns:
        bool: True if valid URL, False otherwise
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


# =============================================
# CORE FUNCTIONS
# =============================================

def extract_products_from_url(url: str) -> list:
    """
    Extract products/services from a given URL using AI.

    Args:
        url (str): Website URL to extract products from

    Returns:
        list: List of extracted product/service names

    Raises:
        Exception: If URL loading or AI processing fails
    """
    # Load web content
    loader = WebBaseLoader(url)
    docs = loader.load()

    if not docs or not docs[0].page_content.strip():
        raise Exception("No content found at the provided URL")

    # Create AI prompt for product extraction
    prompt = ChatPromptTemplate.from_template(
        """
        You are an expert information extractor.  
        Analyze the {page_content} and identify all distinct **products, services, or activities** being offered.  
        
        ⚠️ Rules:  
        - Preserve the full wording exactly as it appears in the text (do not shorten, translate, or rephrase).  
        - Include individual activities/projects if they are presented as offerings (e.g., "Craft your own medieval currency").  
        - Do not include generic descriptions or benefits unless they are framed as a named product/service.  
        
        Return the result strictly in this JSON format:
        {{ "products": ["name1", "name2", "name3"] }}

        Do not add explanations or extra text. Return only valid JSON.
        """
    )

    # Get LLM instance and process with AI
    llm = get_llm_instance()
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    response = chain.invoke({"page_content": docs[0].page_content[:10000]})

    # Clean and parse JSON response
    response = response.strip()
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        response = json_match.group()

    # Parse JSON and extract products
    data = json.loads(response)
    return data.get("products", [])


def generate_facebook_ad_copies(product_name: str) -> str:
    """
    Generate Facebook ad copies for a specific product using AI.

    Args:
        product_name (str): Name of the product/service

    Returns:
        str: Generated ad copies in Markdown format

    Raises:
        Exception: If AI processing fails
    """
    prompt = ChatPromptTemplate.from_template(
        """
        Act as an expert Facebook copywriter with 10+ years of experience creating high-converting ads.

        Generate 10 compelling Facebook ad copies for: {product}

        Requirements:
        - Each copy must be under 50 words
        - Use persuasive, action-oriented language
        - Include relevant emojis where appropriate
        - Make each copy unique and engaging

        Provide 2 copies for each of these 5 proven styles:

        1) **Emotional Appeal** - Connect with feelings, desires, aspirations
        2) **Benefit-Driven** - Focus on specific benefits and solutions
        3) **Urgency/Scarcity** - Create time pressure or limited availability
        4) **Social Proof/Storytelling** - Use testimonials, success stories, or relatable scenarios  
        5) **Curiosity/Question-Based** - Intrigue with questions or surprising facts

        Format as:
        **Style Name:**
        1. [Copy 1]
        2. [Copy 2]

        Focus on conversion-driving elements like clear value propositions, emotional triggers, and strong call-to-actions.
        """
    )

    llm = get_llm_instance()
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    return chain.invoke({"product": product_name}).strip()


# =============================================
# STREAMLIT UI
# =============================================

def main():
    """Main Streamlit application interface."""

    # Page configuration
    st.title("🚀 Product Extractor & Ad Copy Generator")
    st.markdown("Extract products from any website and generate Facebook ad copies using AI")

    # Display current model info
    st.info(get_current_model_info())

    # ==================
    # SIDEBAR - LLM SELECTION & INPUT
    # ==================
    display_llm_selector()

    st.sidebar.header("📥 Website Input")
    user_url = st.sidebar.text_input(
        "Enter Website URL:",
        placeholder="example.com or www.example.com"
    )
    st.sidebar.markdown("*https:// will be added automatically*")

    # Check if LLM is ready before allowing extraction
    if not is_llm_ready():
        st.sidebar.error("❌ Please provide API key to proceed")

    # Extract products button
    extract_button_disabled = not is_llm_ready() or not user_url.strip()

    if st.sidebar.button("🔍 Extract Products", type="primary", disabled=extract_button_disabled):
        if not user_url.strip():
            st.sidebar.warning("Please enter a URL")
            return

        # Process URL
        normalized_url = normalize_url(user_url)

        if not is_valid_url(normalized_url):
            st.error("❌ Please enter a valid URL")
            return

        # Extract products with loading indicator
        with st.spinner(f"🔄 Analyzing {normalized_url}..."):
            try:
                products = extract_products_from_url(normalized_url)

                if products:
                    st.session_state.products = products
                    st.session_state.processed_url = normalized_url
                    st.session_state.ad_copies = ""  # Reset ad copies
                    st.success(f"✅ Found {len(products)} products/services")
                else:
                    st.warning("⚠️ No products/services found on this page")

            except json.JSONDecodeError:
                st.error("❌ Failed to parse AI response. Please try again.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    # ==================
    # MAIN AREA - RESULTS
    # ==================

    if not st.session_state.products:
        # Welcome screen
        st.info("👆 Configure your AI model with API key and enter a website URL in the sidebar to get started")

        # Show setup instructions
        with st.expander("📖 Setup Guide", expanded=True):
            st.markdown("""
            ### Getting Started:
            
            1. **Choose AI Model** (Sidebar):
               - **Google Gemini**: Fast and efficient AI model from Google
               - **OpenAI GPT**: Powerful GPT-3.5-Turbo model from OpenAI
            
            2. **Enter API Key**:
               - **Google Gemini**: Get your key from [Google AI Studio](https://makersuite.google.com/app/apikey)
               - **OpenAI**: Get your key from [OpenAI Platform](https://platform.openai.com/api-keys)
            
            3. **Enter Website URL** and click "Extract Products"
            
            4. **Generate Ad Copies** for any discovered product/service
            
            ### Security:
            - API keys are stored only for your current session
            - Keys are never saved permanently or transmitted elsewhere
            - Your keys remain secure and private
            """)
        return

    # Display results
    st.subheader(f"📦 Results from: {st.session_state.processed_url}")

    # Show extracted products
    with st.expander("📋 View all extracted products", expanded=True):
        for i, product in enumerate(st.session_state.products, 1):
            st.write(f"**{i}.** {product}")

    # Product selection and actions
    selected_product = st.selectbox(
        "🎯 Choose product for ad copy generation:",
        options=st.session_state.products,
        index=0
    )

    # Action buttons
    col1, col2 = st.columns([2, 1])

    with col1:
        generate_button_disabled = not is_llm_ready()
        if st.button("🚀 Generate Facebook Ad Copies", type="primary", disabled=generate_button_disabled):
            if not is_llm_ready():
                st.error("❌ Please configure AI model and API key first")
                return

            with st.spinner(f"✨ Creating ad copies for '{selected_product}'..."):
                try:
                    ad_copies = generate_facebook_ad_copies(selected_product)
                    st.session_state.ad_copies = ad_copies
                    st.success(f"✅ Generated ad copies for: **{selected_product}**")
                except Exception as e:
                    st.error(f"❌ Failed to generate ad copies: {str(e)}")

    with col2:
        if st.button("🗑️ Clear All", type="secondary"):
            st.session_state.products = []
            st.session_state.processed_url = ""
            st.session_state.ad_copies = ""
            st.rerun()

    # Display generated ad copies
    if st.session_state.ad_copies:
        st.markdown("---")
        st.subheader(f"📢 Ad Copies for '{selected_product}'")

        # Layout: Content + Actions
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown(st.session_state.ad_copies)

        with col2:
            st.markdown("### ⚡ Actions")

            # Download button
            download_content = f"Facebook Ad Copies: {selected_product}\n{'=' * 50}\n\n{st.session_state.ad_copies}"
            st.download_button(
                "💾 Download",
                data=download_content,
                file_name=f"{selected_product.replace(' ', '_')}_ads.txt",
                mime="text/plain"
            )

            # Regenerate button
            regenerate_disabled = not is_llm_ready()
            if st.button("🔄 Regenerate", disabled=regenerate_disabled):
                if not is_llm_ready():
                    st.error("❌ Please configure AI model first")
                    return

                with st.spinner("🔄 Regenerating..."):
                    try:
                        new_copies = generate_facebook_ad_copies(selected_product)
                        st.session_state.ad_copies = new_copies
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

            # Clear ads button
            if st.button("❌ Clear Ads"):
                st.session_state.ad_copies = ""
                st.rerun()


# =============================================
# APPLICATION ENTRY POINT
# =============================================

if __name__ == "__main__":
    main()
