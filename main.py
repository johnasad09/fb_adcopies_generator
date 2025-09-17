"""
Product/Service Extractor & Facebook Ad Copy Generator
====================================================

A Streamlit application that:
1. Extracts products/services from any website URL
2. Generates Facebook ad copies for selected products using AI

Dependencies: streamlit, langchain, langchain-google-genai, langchain-community
"""

import streamlit as st
import json
import re
from urllib.parse import urlparse
from secret_key import key
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader

# =============================================
# CONFIGURATION & INITIALIZATION
# =============================================

# Initialize AI model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=key)
output_parser = StrOutputParser()

# Initialize session state
if 'products' not in st.session_state:
    st.session_state.products = []
if 'processed_url' not in st.session_state:
    st.session_state.processed_url = ""
if 'ad_copies' not in st.session_state:
    st.session_state.ad_copies = ""


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
        
        Rules:  
        - Preserve the full wording exactly as it appears in the text (do not shorten, translate, or rephrase).  
        - Include individual activities/projects if they are presented as offerings (e.g., "Craft your own medieval currency").  
        - Do not include generic descriptions or benefits unless they are framed as a named product/service.  
        
        Return the result strictly in this JSON format:
        {{ "products": ["name1", "name2", "name3"] }}

        Do not add explanations or extra text. Return only valid JSON.
        """
    )

    # Process with AI
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
        str: Generated ad copies in markdown format

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

    chain = prompt | llm | output_parser
    return chain.invoke({"product": product_name}).strip()


# =============================================
# STREAMLIT UI
# =============================================

def main():
    """Main Streamlit application interface."""

    # Page configuration
    st.title("Product Extractor & Ad Copy Generator")
    st.markdown("Extract products from any website and generate Facebook ad copies using AI")

    # ==================
    # SIDEBAR - INPUT
    # ==================
    st.sidebar.header("Input")
    user_url = st.sidebar.text_input(
        "Enter Website URL:",
        placeholder="example.com or www.example.com"
    )
    st.sidebar.markdown("*https:// will be added automatically*")

    # Extract products button
    if st.sidebar.button("Extract Products", type="primary"):
        if not user_url.strip():
            st.sidebar.warning("Please enter a URL")
            return

        # Process URL
        normalized_url = normalize_url(user_url)

        if not is_valid_url(normalized_url):
            st.error("Please enter a valid URL")
            return

        # Extract products with loading indicator
        with st.spinner(f"Analyzing {normalized_url}..."):
            try:
                products = extract_products_from_url(normalized_url)

                if products:
                    st.session_state.products = products
                    st.session_state.processed_url = normalized_url
                    st.session_state.ad_copies = ""  # Reset ad copies
                    st.success(f"Found {len(products)} products/services")
                else:
                    st.warning("No products/services found on this page")

            except json.JSONDecodeError:
                st.error("Failed to parse AI response. Please try again.")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    # ==================
    # MAIN AREA - RESULTS
    # ==================

    if not st.session_state.products:
        # Welcome screen
        st.info("Enter a website URL in the sidebar to get started")

        # Example URLs
        # st.subheader("Try these examples:")
        # examples = ["apple.com", "shopify.com", "microsoft.com", "amazon.com"]
        #
        # cols = st.columns(2)
        # for i, url in enumerate(examples):
        #     with cols[i % 2]:
        #         if st.button(f"{url}", key=f"ex_{url}"):
        #             st.info(f"Copy this to sidebar: **{url}**")
        return

    # Display results
    st.subheader(f"Result from: {st.session_state.processed_url}")

    # Show extracted products
    with st.expander("View all", expanded=True):
        for i, product in enumerate(st.session_state.products, 1):
            st.write(f"**{i}.** {product}")

    # Product selection and actions
    selected_product = st.selectbox(
        "Choose product for ad copies:",
        options=st.session_state.products,
        index=0
    )

    # Action buttons
    col1, col2 = st.columns([2, 1])

    with col1:
        if st.button("Generate Facebook Ad Copies", type="primary"):
            with st.spinner(f"Creating ad copies for '{selected_product}'..."):
                try:
                    ad_copies = generate_facebook_ad_copies(selected_product)
                    st.session_state.ad_copies = ad_copies
                    st.success(f"Generated ad copies for: **{selected_product}**")
                except Exception as e:
                    st.error(f"Failed to generate ad copies: {str(e)}")

    with col2:
        if st.button("🗑Clear All", type="secondary"):
            st.session_state.products = []
            st.session_state.processed_url = ""
            st.session_state.ad_copies = ""
            st.rerun()

    # Display generated ad copies
    if st.session_state.ad_copies:
        st.markdown("---")
        st.subheader(f"Ad Copies for '{selected_product}'")

        # Layout: Content + Actions
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown(st.session_state.ad_copies)

        with col2:
            st.markdown("###Actions")

            # Download button
            download_content = f"Facebook Ad Copies: {selected_product}\n{'=' * 50}\n\n{st.session_state.ad_copies}"
            st.download_button(
                "Download",
                data=download_content,
                file_name=f"{selected_product.replace(' ', '_')}_ads.txt",
                mime="text/plain"
            )

            # Regenerate button
            if st.button("Regenerate"):
                with st.spinner("Regenerating..."):
                    try:
                        new_copies = generate_facebook_ad_copies(selected_product)
                        st.session_state.ad_copies = new_copies
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

            # Clear ads button
            if st.button("Clear Ads"):
                st.session_state.ad_copies = ""
                st.rerun()


# =============================================
# APPLICATION ENTRY POINT
# =============================================

if __name__ == "__main__":
    main()
