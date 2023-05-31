import openai
import nltk
from nltk.chat.util import Chat, reflections
from textblob import TextBlob

pairs = [
    [
        r"my name is (.*)",
        ["Hello %1, how can I assist you today?"],
    ],
    [
        r"hi|hello|hey",
        ["Hello, how can I assist you today?"],
    ],
    [
        r"what is your name?",
        ["My name is Chatbot, how can I assist you today?"],
    ],
    [
        r"what can you do?",
        ["I can assist you with any questions or problems you may have. How can I help you today?"],
    ],
    [
        r"bye|goodbye",
        ["Goodbye, have a nice day!"],
    ],
]

chatbot = Chat(pairs, reflections)

def sentiment_analysis(text):
    """
    Perform sentiment analysis on the given text and return the sentiment score
    """
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    return sentiment_score

def handle_unknown_input(user_input):
    """
    Handle unknown user input by suggesting some topics or asking the user to clarify their question
    """
    response = "I'm sorry, I didn't understand your question. Here are some topics you might be interested in: "
    # Add some suggested topics here
    response += "weather, news, sports"
    return response

def generate_response(user_input):
    """
    Generate a response to the user's input using the OpenAI API
    """
    openai.api_key = "YOUR_API_KEY"
    prompt = f"Chat with me: {user_input}"
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.5,
        api_key=openai.api_key
    )
    return response.choices[0].text.strip()

while True:
    user_input = input("You: ")
    sentiment_score = sentiment_analysis(user_input)
    if sentiment_score > 0.5:
        print("Chatbot: That's great to hear!")
    elif sentiment_score < -0.5:
        print("Chatbot: I'm sorry to hear that.")
    else:
        try:
            response = generate_response(user_input)
        except:
            response = handle_unknown_input(user_input)
        print("Chatbot:", response)
