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

nltk.download("stopwords")
nltk.download("punkt")

nlp = spacy.load("en_core_web_sm")

def find_song_and_artist_from_openai(text):
    openai.api_key = st.session_state.api_key
    lines = text.splitlines()
    recommended_songs = []

    song_name = "Unknown Song"
    song_artist = "Unknown Artist"

    for line in lines:
        if line.strip():
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini-2024-07-18",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"You are a professional song lover and listener who could find songs from lyrics efficiently, and you are also know every artists."
                            f"Find what songs these lyrics come from by comparing the lyrics to the songs you know. "
                            f"Find the name of the song and the artist. "
                            f"You could also recommend other 3 from the same artist to recommend users. "
                            f"Recommended songs might be that artists' popular songs or maybe the songs from that artists that share a lot of same word with the lyrics that users input"
                        )
                    },
                    {"role": "user", "content": line}
                ],
                max_tokens=3000
            )
            response_content = response['choices'][0]['message']['content'].strip()
            
            parts = response_content.split(", ")
            
            if len(parts) > 0:
                song_info = parts[0].split(" - ")
                if len(song_info) == 2:
                    song_name = song_info[0].strip()
                    song_artist = song_info[1].strip()

            if len(parts) > 1:
                recommended_songs.extend(parts[1:])

    return song_name, song_artist, recommended_songs

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

# Layout
st.set_page_config(page_title="Lyrics Translator 🎤", page_icon=".streamlit/favicon.ico", layout="wide")

# Sidebar 
st.sidebar.title("API Key")
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

# Store the API key 
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
   song_name = "Unknown Song"
   song_artist = "Unknown Artist"
   recommended_songs = []
   
   # Translate to English
   with col2:
       if st.button("Translate to ENG"):
            target_language = "English"
            if input_text and 'api_key' in st.session_state:
                song_name, song_artist, recommended_songs = find_song_and_artist_from_openai(input_text)

            # Display Song Name and Artist centered
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown(f"""
                <div style="border: 2px solid #4CAF50; padding: 10px; border-radius: 8px; background-color: #fafafa;">
                    <p style='font-size: 16px; color: #333;'>Song Name: {song_name}</p>
                    <p style='font-size: 16px; color: #333;'>by: {song_artist}</p>
                </div>
                """, unsafe_allow_html=True)

            # Display recommended songs in 3 columns
            if recommended_songs:
                st.subheader("Recommended Songs:")

                # Create 3 columns for recommended songs
                col1, col2, col3 = st.columns(3)
                with col1:
                    if len(recommended_songs) > 0:
                        st.text_area(recommended_songs[0], height=100)
                with col2:
                    if len(recommended_songs) > 1:
                        st.text_area(recommended_songs[1], height=100)
                with col3:
                    if len(recommended_songs) > 2:
                        st.text_area(recommended_songs[2], height=100)
            
            # Translate the text
            translated_text = translate_text_with_openai(input_text, target_language)

            # Display translated text
            translated_text_with_br = translated_text.replace("\n", "<br>")
            st.subheader(f"Translated Text to {target_language}")
            st.markdown(f"""
            <div style="border: 2px solid #4CAF50; padding: 10px; border-radius: 8px; background-color: #fafafa;">
                    <p style='font-size: 16px; color: #333;'>{translated_text_with_br}</p>
            </div>
            """, unsafe_allow_html=True)  

            # Summarize the translated text
            summary = generate_summary(translated_text)

            st.subheader("Summary of Lyrics")
            st.markdown("""
            <div style="border: 2px solid #4CAF50; padding: 10px; border-radius: 8px; background-color: #fafafa;">
                <p style='font-size: 16px; color: #444;'>""" + summary + "</p></div>", unsafe_allow_html=True)

            # Generate Word Frequency and Export to Excel
            excel_buffer, word_counts_df, filtered_words = most_common(input_text)

            # Display word frequency as DataFrame
            st.subheader("Top 10 Words")
            excel_buffer, word_counts_df,_ = most_common(input_text)
            st.dataframe(word_counts_df, use_container_width=True)

            # Provide Excel download link
            st.download_button(
                label="see all frequency",
                data=excel_buffer,
                file_name="word_frequency.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

   # Translate to Thai
   with col3:
       if st.button("Translate to THA"):
            target_language = "Thai"
            if input_text and 'api_key' in st.session_state:
                song_name, song_artist, recommended_songs = find_song_and_artist_from_openai(input_text)

            # Display Song Name and Artist centered
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown(f"""
                <div style="border: 2px solid #4CAF50; padding: 10px; border-radius: 8px; background-color: #fafafa;">
                    <p style='font-size: 16px; color: #333;'>Song Name: {song_name}</p>
                    <p style='font-size: 16px; color: #333;'>by: {song_artist}</p>
                </div>
                """, unsafe_allow_html=True)

            # Display recommended songs in 3 columns
            if recommended_songs:
                st.subheader("Recommended Songs:")

                # Create 3 columns for recommended songs
                col1, col2, col3 = st.columns(3)
                with col1:
                    if len(recommended_songs) > 0:
                        st.text_area(recommended_songs[0], height=100)
                with col2:
                    if len(recommended_songs) > 1:
                        st.text_area(recommended_songs[1], height=100)
                with col3:
                    if len(recommended_songs) > 2:
                        st.text_area(recommended_songs[2], height=100)

            # Translate the text
            translated_text = translate_text_with_openai(input_text, target_language)

            # Display translated text
            translated_text_with_br = translated_text.replace("\n", "<br>")
            st.subheader(f"Translated Text to {target_language}")
            st.markdown(f"""
            <div style="border: 2px solid #4CAF50; padding: 10px; border-radius: 8px; background-color: #fafafa;">
                    <p style='font-size: 16px; color: #333;'>{translated_text_with_br}</p>
            </div>
            """, unsafe_allow_html=True)  

            # Summarize the translated text
            summary = generate_summary(translated_text)

            st.subheader("Summary of Lyrics")
            st.markdown("""
            <div style="border: 2px solid #4CAF50; padding: 10px; border-radius: 8px; background-color: #fafafa;">
                <p style='font-size: 16px; color: #444;'>""" + summary + "</p></div>", unsafe_allow_html=True)

            # Generate Word Frequency and Export to Excel
            excel_buffer, word_counts_df, filtered_words = most_common(input_text)

            # Display word frequency as DataFrame
            st.subheader("Top 10 Words")
            excel_buffer, word_counts_df,_ = most_common(input_text)
            st.dataframe(word_counts_df, use_container_width=True)
            
            # Provide Excel download link
            st.download_button(
                label="see all frequency",
                data=excel_buffer,
                file_name="word_frequency.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
