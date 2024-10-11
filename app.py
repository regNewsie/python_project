import gradio as gr
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Load the dataset (Excel sheet with columns 'Problems' and 'Solutions')
df = pd.read_excel('customer_inquiries.xlsx')  # Add your dataset path here

# Function to preprocess user input
def preprocess_input(user_input):
    tokens = word_tokenize(user_input)
    tokens = [token.lower() for token in tokens]
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens

# Function to find the best match solution for the user query
def find_best_match(user_input, df):
    texts = [user_input] + list(df['Problems'])
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    best_match_idx = cosine_sim.argmax()
    return df['Solutions'].iloc[best_match_idx]

# Gradio function that handles user input
def chatbot_response(user_input):
    # Preprocess the user input
    processed_input = preprocess_input(user_input)
    processed_input_str = ' '.join(processed_input)
    
    # Find the best match from the dataset
    response = find_best_match(processed_input_str, df)
    
    return response

# Instruction message to display to the user
instruction_text = "Welcome to the Rice Cooker Support Bot! Type your problem below and the bot will help you with a solution."

# Build the Gradio interface
with gr.Blocks() as interface:
    gr.Markdown(instruction_text)
    
    # Textbox for user input and output for the chatbot's response
    user_input = gr.Textbox(label="Your Problem", placeholder="Describe your rice cooker issue here...")
    chatbot_output = gr.Textbox(label="Solution", interactive=False)
    
    # When user submits the query, this button triggers the chatbot response
    submit_button = gr.Button("Get Solution")
    
    # Link the button with the chatbot response function
    submit_button.click(fn=chatbot_response, inputs=user_input, outputs=chatbot_output)

# Launch the Gradio interface
interface.launch()
