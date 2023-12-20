from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import speech_recognition as sr

audio_file_path = "/home/manish/Desktop/monteage/new_face_code/speeches/file_20231220052415.wav"
recognizer = sr.Recognizer()

with sr.AudioFile(audio_file_path) as source:
    audio_data = recognizer.record(source)

try:
    transcribed_text = recognizer.recognize_google(audio_data)
    sentiment = SentimentIntensityAnalyzer()
    sent_1 = sentiment.polarity_scores(transcribed_text)
    print("Sentiment of text 1:", sent_1)
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")
except sr.RequestError as e:
    print(f"Could not request results from Google Speech Recognition service; {e}")

