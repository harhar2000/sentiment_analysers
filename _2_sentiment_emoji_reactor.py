from textblob import TextBlob
from dataclasses import dataclass

# Define a data class to represent the Mood
@dataclass
class Mood:
    emoji: str
    sentiment: float

# Determine mood based on input text sentiment and threshold
def get_mood(input_text: str, *, threshold: float) -> Mood:
    sentiment: float = TextBlob(input_text).sentiment.polarity  # Use TextBlob, analyse input_text seniment,
                                                                # Give score between -1 and 1
    # Create range for neutral sentiment
    friendly_threshold: float = threshold       
    hostile_threshold: float = -threshold

    # Determine mood based on sentiment score compared to thresholds
    if sentiment >= friendly_threshold:
        return Mood('😊', sentiment)
    elif sentiment <= hostile_threshold:
        return Mood('😠', sentiment)
    else: 
        return Mood('😑', sentiment)
    
if __name__ == "__main__":
    while True:
        text: str = input("Text: ")
        mood: Mood = get_mood(text, threshold=0.3)

        print(f"{mood.emoji} ({mood.sentiment})")