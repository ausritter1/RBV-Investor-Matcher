import streamlit as st
import pandas as pd
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from typing import List
import json
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="RBV Portfolio Company VC Matcher",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load and display RBV logo
logo = Image.open("logo.png")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image(logo, width=150)

# Check for API key in secrets
if 'openai_api_key' not in st.secrets:
    st.error("OpenAI API key not found. Please set it in the Streamlit secrets.")
    st.stop()

# Set OpenAI API key from secrets
os.environ['OPENAI_API_KEY'] = st.secrets['openai_api_key']

# Initialize models
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(temperature=0.1)

def load_and_process_data():
    """Load and process the VC relationships data"""
    df = pd.read_csv("vcrellys.csv")

    # Create documents for vectorstore
    documents = []
    for _, row in df.iterrows():
        content = f"""
        Firm: {row['Company']}
        Domain: {row['Domain']}
        Description: {row['Description']}
        Preferred Stage: {row['Preferred Stage']}
        Preferred Sector: {row['Preferred Sector']}
        """
        documents.append({"content": content, "metadata": dict(row)})

    return df, documents

def create_vectorstore(documents):
    """Create FAISS vectorstore from documents"""
    texts = [doc["content"] for doc in documents]
    metadatas = [doc["metadata"] for doc in documents]
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    return vectorstore

def analyze_startup(description: str, vectorstore) -> dict:
    """Analyze startup and find matching investors"""
    template = """You are an expert VC analyst for RBV. Analyze the following portfolio company description and recommend the most suitable investors from our network.

    Portfolio Company Description: {description}

    Based on the provided context about investors in our network, identify the top 3 most relevant investors and explain why they would be good matches. 
    Pay special attention to matching:
    1. The company's stage with the investor's preferred stage
    2. The company's sector with the investor's preferred sectors
    3. Any domain-specific alignment
    
    Context about potential investors:
    {context}

    Please provide your response in the following JSON format:
    {{
        "matches": [
            {{
                "investor": "Name of investor firm",
                "rationale": "2-3 sentence explanation of why this investor is a good match, specifically mentioning stage and sector fit"
            }}
        ],
        "summary": "2-3 sentence overall summary of the matching logic, highlighting stage and sector alignment"
    }}
    """

    prompt = PromptTemplate.from_template(template)
    docs = vectorstore.similarity_search(description, k=5)
    context = "\n\n".join(d.page_content for d in docs)
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"description": description, "context": context})

    return json.loads(result)

# Main app
st.title("RBV Portfolio Company VC Matcher 🤝")
st.write("Match your portfolio companies with the most relevant VCs in our network.")

# Load data
with st.spinner("Loading investor database..."):
    df, documents = load_and_process_data()
    vectorstore = create_vectorstore(documents)

# Input section
st.subheader("Portfolio Company Information")
description = st.text_area(
    "Enter portfolio company description",
    height=150,
    placeholder="Describe the company, including: sector, stage, traction, team background, and any key achievements..."
)

if st.button("Find Matching Investors"):
    if description:
        with st.spinner("Analyzing company and finding matches..."):
            results = analyze_startup(description, vectorstore)

            # Display results
            st.subheader("🎯 Top Investor Matches")

            # Summary
            st.info(results["summary"])

            # Matches
            for i, match in enumerate(results["matches"], 1):
                with st.expander(f"#{i} - {match['investor']}", expanded=True):
                    st.write("**Why this investor?**")
                    st.write(match["rationale"])

                    # Get additional investor info from DataFrame
                    matching_investors = df[df["Company"] == match["investor"]]
                    if not matching_investors.empty:
                        investor_info = matching_investors.iloc[0]

                        cols = st.columns(2)
                        with cols[0]:
                            st.write("**Domain & Description:**")
                            st.write(f"**Domain:** {investor_info['Domain']}")
                            st.write(f"**Description:** {investor_info['Description']}")
                        with cols[1]:
                            st.write("**Investment Preferences:**")
                            st.write(f"**Preferred Stage:** {investor_info['Preferred Stage']}")
                            st.write(f"**Preferred Sector:** {investor_info['Preferred Sector']}")

            # Export options
            st.download_button(
                "Export Results (JSON)",
                data=json.dumps(results, indent=2),
                file_name="investor_matches.json",
                mime="application/json"
            )
    else:
        st.warning("Please enter a portfolio company description first.")

# Sidebar
with st.sidebar:
    st.subheader("About")
    st.write("""
    This tool helps match RBV portfolio companies with the most relevant investors from our network.

    It uses:
    - Advanced AI matching
    - Deep investor network analysis
    - Contextual understanding

    To get the best results:
    1. Provide detailed company information
    2. Include current stage and traction
    3. Highlight unique technology or advantages
    4. Mention target market and growth plans
    """)

    st.markdown("---")
    st.caption("Read Beard Ventures © 2025")
