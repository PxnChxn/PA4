#Lyrics Translator

##Key Features:
-  Song Lyric Translation: Translate lyrics from one language to another (either Thai or English), while maintaining their emotional impact and artistic quality.
- Word-Level Translation: View word translations in a table, along with their corresponding parts of speech. You can also download the results in an Excel file.
- Summary Generation: Summarize the translated lyrics, retaining key messages and emotional tones.
- Word Frequency Analysis: Explore the most common words in the input text and download a word frequency report.
- Song Lyric Chatbot: Engage with a chatbot that discusses song lyrics, providing deeper insights into their meaning and themes.

##Getting Started
You can access the app through the following link:
[Lyrics Translator App](https://translatelyricsproject.streamlit.app/)

###To start using the app, simply follow these steps:

1. Enter your OpenAI API Key: You will need an openai API key to access the translation features. Enter it in the sidebar to begin using the app.
2. Paste your Lyrics: In the input area, paste the song lyrics you want to translate.
3. Translate the Lyrics: Click the buttons to translate the lyrics into your target language (English or Thai).
4. View the Results: After translation, the app will display the translated lyrics, a word-by-word translation table, a summary of the lyrics, and a word frequency analysis.
5. Interact with the Chatbot: Ask questions about the song lyrics, and the chatbot will provide insights based on the input text and the translation.

##Modules Used
langdetect      | nltk  
openpyxl        | OpenAI GPT Models  
pandas          | PyThaiNLP  
re              | spaCy  
Streamlit       | Python  

##How It Works
**Input Text**: The user inputs song lyrics in either Thai or English.
**Translation**: The app uses OpenAI's models to translate the lyrics into the target language, preserving the artistic qualities of the original song.
**Word-Level Translation**: The app generates a word-by-word translation and displays it in a table.
**Summary**: The app generates a summary of the translated lyrics, capturing the key message and emotional tone.
**Word Frequency Analysis**: It analyzes the input text to identify the most common words, providing a downloadable word frequency report.
**Chatbot**: The chatbot allows the user to engage in a conversation about the song lyrics, providing insights based on the translation.


##Developers:
- Panus Prasongsamret
- Chananya Sunthonphrao
