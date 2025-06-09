import streamlit as st
import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

# --- Set your Hugging Face API Token ---
# IMPORTANT: Replace "hf_YOUR_API_TOKEN" with your actual token.
# For production, set this as an environment variable (e.g., in your shell or Streamlit Cloud secrets)
# export HUGGINGFACEHUB_API_TOKEN="hf_YOUR_API_TOKEN"
# Or directly in your script (less secure for sharing/production):



# Replace this block in your app.py
# if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
#     st.error("HUGGINGFACEHUB_API_TOKEN environment variable not set.")
#     st.markdown("Please set your Hugging Face API token as an environment variable or directly in the script.")
#     st.stop() # Stop the app if the token is missing

# Use st.secrets to securely get the API token
if "HUGGINGFACEHUB_API_TOKEN" not in st.secrets:
    st.error("HUGGINGFACEHUB_API_TOKEN not found in Streamlit secrets.")
    st.markdown("Please add your Hugging Face API token to your Streamlit Cloud secrets.")
    st.stop()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

@st.cache_resource
def setup_langchain_models():
    common_model_kwargs = {
        "temperature": 0.7,       
        "max_new_tokens": 512,    
        "return_full_text": False  
    }

    llm1 = HuggingFaceEndpoint(
        repo_id='google/gemma-2-2b-it',
        task='text-generation'
    )

    llm2 = HuggingFaceEndpoint(
        repo_id='mistralai/Mistral-Small-3.1-24B-Instruct-2503',
        task='text-generation'
    )

 
    model1 = ChatHuggingFace(llm=llm1, repo_id='google/gemma-2-2b-it')
    model2 = ChatHuggingFace(llm=llm2, repo_id='mistralai/Mistral-Small-3.1-24B-Instruct-2503')

    prompt1 = PromptTemplate(
        template='Generate short and simple notes from the following text:\n\nTEXT: {text}\n\nNOTES:',
        input_variables=['text']
    )

    prompt2 = PromptTemplate(
        template='Generate 5 short questions from the following text:\n\nTEXT: {text}\n\nQUESTIONS:',
        input_variables=['text']
    )

    prompt3 = PromptTemplate(
        template='Merge the provided notes and quiz into a single coherent document. Ensure clear separation between notes and quiz sections.\n\nNOTES:\n{notes}\n\nQUIZ:\n{quiz}\n\nMERGED DOCUMENT:',
        input_variables=['notes', 'quiz']
    )

    parser = StrOutputParser()


    notes_chain = prompt1 | model1 | parser
    quiz_chain = prompt2 | model2 | parser


    parallel_execution_for_merge = RunnableParallel({
        'notes': notes_chain,
        'quiz': quiz_chain
    })

    
    merge_chain = prompt3 | model1 | parser

    
    final_chain = parallel_execution_for_merge | merge_chain
    
    
    return notes_chain, quiz_chain, final_chain

# Setup the chains once when the app starts
notes_chain_instance, quiz_chain_instance, final_chain_instance = setup_langchain_models()


st.title("Text to Notes & Quiz Generator")
st.markdown("Enter your text below to generate concise notes, a short quiz, and a merged document.")


input_text = st.text_area("Paste your text here:", height=300)

if st.button("Generate Notes & Quiz"):
    if input_text:
        with st.spinner("Generating... This might take a moment as LLMs are working."):
            try:
                notes_output = notes_chain_instance.invoke({'text': input_text})
                st.markdown(notes_output)

                quiz_output = quiz_chain_instance.invoke({'text': input_text})
                st.markdown(quiz_output)

                merged_output = final_chain_instance.invoke({'text': input_text})
                st.markdown(merged_output)

            except Exception as e:
                st.error(f"An error occurred during generation: {e}")
                st.markdown("Please ensure your `HUGGINGFACEHUB_API_TOKEN` is correctly set and the models are accessible.")
                st.markdown("You can get a token from [Hugging Face settings](https://huggingface.co/settings/tokens).")
    else:
        st.warning("Please enter some text to generate notes and quiz.")

st.info("Powered by LangChain and Hugging Face LLMs (Gemma and Mistral).")