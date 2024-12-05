import pythainlp.corpus
import streamlit as st
import openai
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
import pythainlp
from pythainlp.tokenize import word_tokenize as thai_tokenize
import pandas as pd
import openpyxl
import io
from io import BytesIO
from langdetect import detect
import spacy
from textblob import TextBlob
from transformers import BertTokenizer, BertForSequenceClassification
import torch

nltk.download("stopwords")
nltk.download("punkt")

nlp = spacy.load("en_core_web_sm")

# Function to call OpenAI API for translation
def translate_text_with_openai(text, target_language):
    openai.api_key = st.session_state.api_key
    
    lines = text.splitlines()  
    translated_lines = []

    for line in lines:
        if line.strip(): 
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini-2024-07-18",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"You are a professional song lyric translator with expertise in preserving both the meaning and the artistic qualities of lyrics. "
                            f"Translate the following song line into {target_language}. "
                            f"While translating, make sure to preserve the emotional tone, rhythm, and poetic essence of the original text. "
                            f"Adapt cultural references and idiomatic expressions to be relevant and natural in {target_language}. "
                            f"Avoid literal translations; instead, focus on capturing the meaning, mood, and key message of the song. "
                            f"Pay attention to any figurative language, metaphor, or wordplay and translate them in a way that fits the target language and retains the artistic intent. "
                            f"Ensure the translation flows naturally, keeping in mind that song lyrics may require slight adjustments to fit melody and cadence. "
                            f"Do not ignore any parentheses or special formatting in the original lyrics; if any exist, preserve them as they are."
                        )
                    },
                    {"role": "user", "content": line}
                ],
                max_tokens=3000
            )
            translated_line = response.choices[0].message["content"].strip()

            translated_lines.append(translated_line)
        else:
            translated_lines.append("") 

    translated_text = "\n".join(translated_lines)
    return translated_text

def generate_summary(translate_result):
    detected_language = detect(translate_result)  

    if detected_language != 'th':
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Summarize this text: {translate_result}. Make it still contain the important key message from the lyric."}
            ],
            max_tokens=300
        )
        summary = response['choices'][0]['message']['content'].strip()
    else:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"สรุปเนื้อหาของข้อความนี้: {translate_result}. ทำให้ยังคงความหมายสำคัญจากเนื้อเพลง."}
            ],
            max_tokens=300
        )
        summary = response['choices'][0]['message']['content'].strip()
        
    return summary
    
def get_stopwords():
   english_stopwords = set(stopwords.words('english'))
   thai_stopwords = set(pythainlp.corpus.thai_stopwords())
   stopwords_combined = english_stopwords.union(thai_stopwords)
   stopwords_combined.add(' ')
   stopwords_combined.add('\n')
   
   return stopwords_combined

def tokenization(input_text):
   if re.match(r"[ก-์๐-๙]+", input_text): 
       tokens = pythainlp.tokenize.word_tokenize(input_text)
   else:  
       doc = nlp(input_text)
       tokens = [token.text for token in doc]
   tokens = [token for token in tokens if token.strip() and not re.match(r'[^\w\s]', token)]
   return tokens

def most_common(input_text):
    detected_language = detect(input_text)

    if detected_language == 'th':
        words = pythainlp.tokenize.word_tokenize(input_text)
    else:
        words = tokenization(input_text)

    stopwords_combined = get_stopwords()
    filtered_words = [word for word in words if word not in stopwords_combined]

    word_counts = Counter(filtered_words)
    word_counts_sorted = word_counts.most_common(10)

    word_counts_df = pd.DataFrame(word_counts_sorted, columns=["Word", "Frequency"], index=range(1, len(word_counts_sorted) + 1))

    excel_buffer = io.BytesIO()
    word_all_counts_sorted = word_counts.most_common()
    word_all_counts_df = pd.DataFrame(word_all_counts_sorted, columns=["Word", "Frequency"])

    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        word_all_counts_df.to_excel(writer, index=True, sheet_name="Word Frequency")

    excel_buffer.seek(0)

    return excel_buffer, word_counts_df, filtered_words

def chatbot_response(user_input, conversation_history, translated_text=None, summary=None):
    openai.api_key = st.session_state.api_key
    
    conversation_history.append(f"User: {user_input}")
    
    if input_text:
        conversation_history.append(f"Lyrics: {input_text}")
    if translated_text:
        conversation_history.append(f"Translated Text: {translated_text}")
    if summary:
        conversation_history.append(f"Summary: {summary}")

    prompt = "\n".join(conversation_history) + "\nChatbot (about song lyrics):"
    
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini-2024-07-18", 
        messages=[
            {"role": "system", "content": 
                "You are a helpful chatbot that specializes in discussing song lyrics. "
                "Your task is to engage in conversations based on the lyrics that the user provides, {input_text}"
                "giving meaningful and relevant responses while focusing on the themes, emotions, and messages within the lyrics."},
            {"role": "user", "content": user_input}
        ],
        max_tokens=350,
        n=1, 
        stop=None, 
        temperature=0.7  
    )
    
    bot_reply = response['choices'][0]['message']['content'].strip()
    conversation_history.append(f"Chatbot: {bot_reply}")
    
    return bot_reply, conversation_history

# Layout
st.set_page_config(page_title="Lyrics Translator 🎤", page_icon=".streamlit/favicon.ico", layout="wide")

# Sidebar 
st.sidebar.title("API Key")
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

if api_key:
    st.session_state.api_key = api_key 

# Header
st.title("Translate the Lyrics ✨")

# Main content container
with st.container():
   # Input area
   st.subheader("💿💿 Paste the text to start translation 💿💿")
   input_text = st.text_area("Enter your text here...", height=250)

   # Centered translation buttons
   col1, col2, col3, col4 = st.columns(4)

   # Initialize variables
   translated_text = None
   summary = None
   word_cloud_image = None
   excel_buffer = None

   # Translate to English
   with col2:
       if st.button("Translate to ENG"):
           target_language = "English"
           if input_text and 'api_key' in st.session_state:
               # Translate the text
               translated_text = translate_text_with_openai(input_text, target_language)
               
               # Generate analyses
               summary = generate_summary(translated_text)
               most_common_result = most_common(input_text)
               
               # Store results in session state
               st.session_state.translated_text = translated_text
               st.session_state.most_common = most_common_result
               
   # Translate to Thai
   with col3:
       if st.button("Translate to THA"):
           target_language = "Thai"
           if input_text and 'api_key' in st.session_state:
               # Translate the text
               translated_text = translate_text_with_openai(input_text, target_language)
               
               # Generate analyses
               summary = generate_summary(translated_text)
               most_common_result = most_common(input_text)
               
               # Store results in session state
               st.session_state.translated_text = translated_text
               st.session_state.most_common = most_common_result
               
if 'api_key' not in st.session_state:
   st.warning("Please enter your OpenAI API key in the sidebar.")
elif not input_text:
   st.warning("Please enter some text for translation.")
# Store the current input in session_state to detect changes
if 'previous_input' not in st.session_state:
    st.session_state.previous_input = ""

if 'translated_text' not in st.session_state:
    st.session_state.translated_text = None

if 'summary' not in st.session_state:
    st.session_state.summary = None

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Check if input_text has changed
if input_text != st.session_state.previous_input:
    # Reset the relevant session state variables
    st.session_state.translated_text = None
    st.session_state.summary = None
    st.session_state.conversation_history = []
    st.session_state.previous_input = input_text

if input_text and not st.session_state.translated_text:
    st.session_state.translated_text = translate_text_with_openai(input_text, target_language="English")
    st.session_state.summary = generate_summary(st.session_state.translated_text)

# Display results only if translation has been done
if st.session_state.translated_text:
    translated_text_with_br = st.session_state.translated_text.replace("\n", "<br>")
    
    st.subheader("Translated Text:")
    st.markdown(f"""
    <div style="border: 2px solid #4CAF50; padding: 10px; border-radius: 8px; background-color: #fafafa;">
        <p style='font-size: 16px; color: #333;'>{translated_text_with_br}</p>
    </div>
    """, unsafe_allow_html=True)
   
    st.subheader("Summary:")
    st.markdown(f"""
    <div style="border: 2px solid #4CAF50; padding: 10px; border-radius: 8px; background-color: #fafafa;">
      <p style='font-size: 16px; color: #444;'>{summary}</p>
    </div>
    """, unsafe_allow_html=True)

    # Display the top 10 words table
    st.subheader("Top 10 Words:")
    excel_buffer, word_counts_df, _ = most_common(input_text)
    st.dataframe(word_counts_df, use_container_width=True)
   
    # Download button for word frequency Excel file
    if excel_buffer:
        st.download_button(
            label="See all word frequency",
            data=excel_buffer,
            file_name="word_frequency.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="word_frequency_download"
        )

    # Chat interface
    st.title("Song Lyric Chatbot")
    st.write("Chatbot will discuss the lyrics with you. Keep the conversation going until you're satisfied.")
    
    user_input = st.text_area("Ask about the song:")
    
    submit_button = st.button("Submit")
    
    if submit_button and user_input:
        bot_response, updated_history = chatbot_response(user_input, st.session_state.conversation_history)
        
        st.session_state.conversation_history = updated_history
        
        for message in st.session_state.conversation_history:
            if "User:" in message:
                st.markdown(f"**User:** {message.replace('User: ', '')}")
            elif "Chatbot:" in message:
                st.markdown(f"**Chatbot:** {message.replace('Chatbot: ', '')}")

            elif "Chatbot:" in message:
                st.markdown(f"**Chatbot:** {message.replace('Chatbot: ', '')}")
