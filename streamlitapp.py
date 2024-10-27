import streamlit as st
import os
import json
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from src.mcqgenerator.utils import read_file, get_table_data
import traceback

# Page configuration
st.set_page_config(
    page_title="MCQ Generator",
    page_icon="📚",
    layout="wide"
)

# Title and description
st.title("📚 MCQ Generator App")
st.markdown("""
Generate multiple choice questions from your PDF or text files using AI.
""")

# Sidebar for inputs
with st.sidebar:
    st.header("Configuration")
    
    # File upload
    uploaded_file = st.file_uploader("Upload your PDF or text file", type=['pdf', 'txt'])
    
    # MCQ Parameters
    mcq_count = st.number_input("Number of MCQs", min_value=1, max_value=10, value=3)
    
    subject = st.selectbox(
        "Select Subject",
        ["Machine Learning", "Science", "Mathematics", "English", "History", "Geography", "Computer Science"]
    )
    
    tone = st.selectbox(
        "Select Tone",
        ["Easy", "Medium", "Hard"]
    )
    
    if uploaded_file and st.button("Generate MCQs"):
        with st.spinner("Reading file content..."):
            text = read_file(uploaded_file)
            st.session_state.text = text
            st.session_state.mcq_count = mcq_count
            st.session_state.subject = subject
            st.session_state.tone = tone
            st.session_state.generate = True
    
# Main content area
if 'generate' in st.session_state and st.session_state.generate:
    # Display file content
    with st.expander("Show File Content"):
        st.write(st.session_state.text)
    
    with st.spinner("Generating MCQs..."):
        try:
            # Get API Key from environment variable
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                st.error("OpenAI API key not found. Please check your environment variables.")
                st.stop()
            
            # Initialize the language model
            llm = ChatOpenAI(
                openai_api_key=api_key,
                model_name='gpt-3.5-turbo',  # Changed from gpt-4 to gpt-3.5-turbo for wider accessibility
                temperature=0.5
            )
            
            # Import and set up chains
            from  src.mcqgenerator.mcqgenerator import generate_evaluate_chain
            
            # Generate MCQs
            with get_openai_callback() as cb:
                response = generate_evaluate_chain(
                    {
                        "text": st.session_state.text,
                        "number": st.session_state.mcq_count,
                        "subject": st.session_state.subject,
                        "tone": st.session_state.tone,
                        "response_json": json.dumps(RESPONSE_JSON)
                    }
                )
            
            # Extract and display MCQs
            if response:
                quiz = response['quiz']
                review = response['review']
                
                # Display the review
                st.header("Quiz Analysis")
                st.write(review)
                
                # Convert quiz to table format
                table_data = get_table_data(quiz)
                if table_data:
                    st.header("Generated MCQs")
                    df = pd.DataFrame(table_data)
                    st.table(df)
                    
                    # Download options
                    csv = df.to_csv(index=False).encode('utf-8')
                    json_str = quiz
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            "Download MCQs as CSV",
                            csv,
                            "mcqs.csv",
                            "text/csv",
                            key='download-csv'
                        )
                    with col2:
                        st.download_button(
                            "Download MCQs as JSON",
                            json_str,
                            "mcqs.json",
                            "application/json",
                            key='download-json'
                        )
                    
                # Display token usage
                st.sidebar.header("Token Usage")
                st.sidebar.write(f"Total Tokens: {cb.total_tokens}")
                st.sidebar.write(f"Prompt Tokens: {cb.prompt_tokens}")
                st.sidebar.write(f"Completion Tokens: {cb.completion_tokens}")
                st.sidebar.write(f"Total Cost (USD): ${cb.total_cost:.4f}")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error(traceback.format_exc())

# Reset button
if st.sidebar.button("Reset"):
    st.session_state.clear()
    st.experimental_rerun()

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit and OpenAI")