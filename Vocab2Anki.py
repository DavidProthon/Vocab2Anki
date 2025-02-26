import csv
import sqlite3
import subprocess
import sys
from collections import Counter
from pathlib import Path

import warnings

import contractions
import nltk
import pdfplumber
import spacy
from epub2txt import epub2txt
from wordfreq import top_n_list

# Number of most frequently occurring English words to extract from the document.
# The words in the document are first identified, sorted by frequency,
# and then the top `number_of_user_words` most common words are selected.
number_of_user_words = 1200


# The user selects a difficulty range for the extracted words.
# Choose between: "A1", "A2", "B1", "B2", "C1", "C2", "C2+"
lower_level = "B2"  # Minimum difficulty level
upper_level = "B2" # Maximum difficulty level


# This ensures that only words within the selected difficulty range are included.
# Example:
# If the user sets:
# number_of_user_words = 500
# lower_level = "B1"
# upper_level = "B2"
# 
# The program will extract the 500 most frequently occurring words in the document 
# that fall within the difficulty levels B1 to B2 (inclusive).


# Specifies which pronunciation variant to use for the extracted English words.
# Choose between: "US" (American pronunciation) or "UK" (British pronunciation).
pronunciation = "US"

A1_WORDS = 500  # 500 real words
A2_WORDS = 500  # 500 real words
B1_WORDS = 1000 # 1000 real words
B2_WORDS = 2000 # 2000 real words
C1_WORDS = 4000 # 4000 real words
C2_WORDS = 8000 # 8000 real words

warnings.filterwarnings("ignore", category=FutureWarning, module="ebooklib.epub")

def extract_text_from_pdf(file_path: Path) -> str: 
    extracted_text = ""

    with pdfplumber.open(file_path) as pdf_document:
        for page in pdf_document.pages:
            page_text = page.extract_text()
            extracted_text += page_text

    return extracted_text.strip()

def extract_text_from_epub(file_path: Path) -> str:
    extracted_text = epub2txt(file_path)

    return extracted_text

def load_spacy_model(): 
    """
    Checks if the 'en_core_web_sm' model is downloaded, and if not, it downloads it.
    Returns the loaded spaCy model.
    """
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Model 'en_core_web_sm' is not installed. Downloading...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")

    return nlp

def lemmatize_words(nlp, text: str = False, input_list: list = False) -> list:
    """
    This function processes large text in chunks to avoid memory issues. It splits the input text into smaller 
    parts, each with a size defined by the `chunk_size` variable. Each chunk is processed using 
    the provided spaCy NLP model (`nlp`), and lemmatization is applied to all tokens. Punctuation tokens are 
    excluded from the result, and only the lemmatized forms of the words are added to the output list.
    """

    if input_list:
        text = " ".join(input_list)

    chunk_size = 200000
    lemmatized_words = []

    for i in range(int(len(text)/chunk_size)+1):
        doc = nlp(text[i*chunk_size:(i+1)*chunk_size])
        for token in doc:
            if not token.is_punct:
                lemmatized_words.append(token.lemma_)
    
    return lemmatized_words

def filter_vocabulary_lst(list_of_words: list) -> list: 
    """
    Removes words that are not part of the English vocabulary from the list.
    Specifically filters out words that are not in the NLTK English word list.
    """

    official_english_words = set(nltk.corpus.words.words())
    valid_words = [word for word in list_of_words if word in official_english_words]

    return valid_words

def filter_and_lemmatize_common_words(nlp, language: str = 'en', top_n: int = (A1_WORDS+A2_WORDS+B1_WORDS+B2_WORDS+C1_WORDS+C2_WORDS)*3) -> list: 
    """
    Generates a filtered list of the most common words for a given language.
    
    Steps:
    1. Retrieves the top N most common words.
    2. Lemmatizes the words using the provided NLP model.
    3. Removes duplicate lemmatized words.
    4. Removes words that are not part of the English vocabulary
    """

    most_common_words = list(top_n_list(language, top_n))
    lemmatized_words = lemmatize_words(nlp,input_list = most_common_words)
    unique_words = list(dict.fromkeys(lemmatized_words))
    final_word_list = filter_vocabulary_lst(unique_words)

    return final_word_list

def normalize_text(text: str) -> list: 
    """
    This function takes a text and processes it by:
    - Converting it to lowercase.
    - Expanding contractions (e.g., "don't" -> "do not").
    """

    lowercase_text =text.lower()
    contractions_text = contractions.fix(lowercase_text)

    return contractions_text

def get_word_frequencies(word_list: list) -> dict: 
    """
    Counts the occurrences of each word in the list and returns a sorted dictionary 
    of word frequencies, ordered by frequency in descending order.
    """

    count_dict = dict(Counter(word_list))
    sorted_counts = dict(sorted(count_dict.items(), key=lambda item: item[1], reverse=True))

    return sorted_counts

def filter_vocabulary_dct(word_count_dict : dict) -> dict: 
    """
    Removes words that are not part of the English vocabulary from the dictionary.
    Specifically filters out words that are not in the NLTK English word list.
    """

    official_english_words = set(nltk.corpus.words.words())

    for key in list(word_count_dict.keys()):
        if key not in official_english_words:
            del word_count_dict [key]

    return word_count_dict 

def determine_word_level(word: str,word_levels: dict) -> str: 
    """
    Determines the language level of a word based on its position in the list of most common words.
    """
    for level, words in word_levels.items():
        if word in words:
            return level
        
    return "C2+"

def assign_difficulty_level(word_dict: dict,word_levels: dict) -> dict: 
    """
    This function takes a dictionary of words and assigns a difficulty level to each word
    """

    copied_dict_of_words = word_dict.copy()
    for word in copied_dict_of_words.keys():
        copied_dict_of_words[word] = determine_word_level(word,word_levels)

    return copied_dict_of_words

def get_user_levels(lower_level,upper_level):
    levels = ["A1","A2","B1","B2","C1","C2","C2+"]
    user_choose = []

    lower_index = levels.index(lower_level)
    upper_index = levels.index(upper_level)
    user_choose = levels[lower_index:upper_index + 1]

    return user_choose

def get_user_words(number_of_user_words,words_with_difficulty_levels,user_levels):
    '''
    Selects all levels between the given lower and upper levels (inclusive).
    '''

    user_words = []
    number_found_words = 0

    for key, value in words_with_difficulty_levels.items():
        if number_found_words < number_of_user_words:
            if value in user_levels:
                number_found_words +=1
                user_words.append(key)
        else:
            break
            
    return user_words

def retrieve_word_data(user_words, region ,database_path='data.db'):
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row  
    cursor = connection.cursor()
    pronunciation_column = f"pronunciation - {region}"

    extended_words = []
        
    for word in user_words:
        cursor.execute(f"""
            SELECT word, "{pronunciation_column}", translation 
            FROM dictionary 
            WHERE word = ?
        """, (word,))
        
        result = cursor.fetchone()
        
        if result:
            formatted_word = f"{result['word']} [{result[pronunciation_column]}]"
            extended_words.append([result["translation"], formatted_word])
    
    connection.close()
    
    return extended_words

"""
def get_phonetic_transcription(translate_words):
    result = []
    phonemizer = Phonemizer.from_checkpoint("latin_ipa_forward.pt")
    for cz_word, en_word in translate_words:
        pronunciation = phonemizer(en_word, lang="en_us")
        if pronunciation:
            combined = f"{en_word} [{pronunciation}]"
            result.append((cz_word, combined))
        else:
            result.append((cz_word, en_word))
    
    return result
"""

def words_to_csv(data):
    with open("ANKI_soubor.csv", mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(data)

def main():
    nlp = load_spacy_model()
    nltk.download("words")
    folder_path = Path("dokument")

    filtered_common_words = filter_and_lemmatize_common_words(nlp)

    word_levels = {
        'A1': filtered_common_words[0:A1_WORDS],
        'A2': filtered_common_words[A1_WORDS:A2_WORDS + A1_WORDS],
        'B1': filtered_common_words[A1_WORDS + A2_WORDS:A1_WORDS + A2_WORDS + B1_WORDS],
        'B2': filtered_common_words[A1_WORDS + A2_WORDS + B1_WORDS:A1_WORDS + A2_WORDS + B1_WORDS + B2_WORDS],
        'C1': filtered_common_words[A1_WORDS + A2_WORDS + B1_WORDS + B2_WORDS:A1_WORDS + A2_WORDS + B1_WORDS + B2_WORDS + C1_WORDS],
        'C2': filtered_common_words[A1_WORDS + A2_WORDS + B1_WORDS + B2_WORDS + C1_WORDS:A1_WORDS + A2_WORDS + B1_WORDS + B2_WORDS + C1_WORDS + C2_WORDS]
    }


    for file in folder_path.iterdir():
        if file.suffix == ".pdf":
            text = extract_text_from_pdf(file)
        elif file.suffix == ".epub":
            text = extract_text_from_epub(file)

        normalized_text_tokens = normalize_text(text) 
        lemmatized_words = lemmatize_words(nlp,text = normalized_text_tokens)
        word_frequencies = get_word_frequencies(lemmatized_words) 
        filtered_frequencies = filter_vocabulary_dct(word_frequencies) 

        words_with_difficulty_levels = assign_difficulty_level(filtered_frequencies,word_levels)
        user_levels = get_user_levels(lower_level,upper_level)
        user_words = get_user_words(number_of_user_words,words_with_difficulty_levels,user_levels)

        all_necessary_data = retrieve_word_data(user_words,pronunciation)
        #print(all_necessary_data)

        words_to_csv(all_necessary_data)

if __name__ == "__main__":
    main()