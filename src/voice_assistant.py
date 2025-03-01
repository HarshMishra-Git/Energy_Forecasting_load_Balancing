import speech_recognition as sr
from transformers import pipeline

class VoiceAssistant:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.nlp_model = pipeline("text-generation", model="gpt2")

    def activate(self):
        with sr.Microphone() as source:
            print("Listening...")
            audio = self.recognizer.listen(source)

        try:
            query = self.recognizer.recognize_google(audio)
            print(f"User said: {query}")
            return query
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            return None

    def respond_to_query(self, query):
        response = self.nlp_model(query, max_length=50, num_return_sequences=1)
        return response[0]['generated_text']