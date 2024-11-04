import os
import sys
import queue
import sounddevice as sd
import json
from vosk import Model, KaldiRecognizer

# Replace with the path to your downloaded model
MODEL_PATH = "vosk-model-small-en-us-0.15"

# List of preselected phrases/words to detect
PRESELECTED_PHRASES = [
    "a",
    "a sharp",
    "b",
    "c",
    "c sharp",
    "d",
    "d sharp",
    "e",
    "f",
    "f sharp",
    "g",
    "g sharp"
]

# Sort phrases by length in descending order to prioritise longer phrases
PRESELECTED_PHRASES = sorted(PRESELECTED_PHRASES, key=lambda x: len(x.split()), reverse=True)

# Initialise the model
if not os.path.exists(MODEL_PATH):
    print(
        f"Please download the model from https://alphacephei.com/vosk/models and unpack as {MODEL_PATH} in the current folder.")
    sys.exit(1)

model = Model(MODEL_PATH)

# Convert phrases to JSON format for grammar
grammar = json.dumps(PRESELECTED_PHRASES)

recogniser = KaldiRecognizer(model, 16000, grammar)
recogniser.SetWords(True)

# Queue to hold audio data
q = queue.Queue()


def callback(indata, frames, time, status):
    """This callback is called for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))


def calculate_phrase_confidence(phrase_words, recognised_words):
    """
    Calculate the average confidence for the detected phrase.

    :param phrase_words: List of words in the phrase to detect.
    :param recognised_words: List of recognised words with confidences.
    :return: Average confidence score of the phrase.
    """
    confidences = []
    phrase_length = len(phrase_words)
    for i in range(len(recognised_words) - phrase_length + 1):
        # Extract a slice of words to compare with the phrase
        slice_words = [word_info['word'] for word_info in recognised_words[i:i + phrase_length]]
        if slice_words == phrase_words:
            # Extract confidences for these words
            slice_confidences = [word_info['conf'] for word_info in recognised_words[i:i + phrase_length]]
            average_conf = sum(slice_confidences) / phrase_length
            confidences.append(average_conf)
    if confidences:
        return max(confidences)  # Return the highest confidence found for the phrase
    else:
        return None


def main():
    print("Listening... Press Ctrl+C to stop.")
    try:
        with sd.RawInputStream(samplerate=16000, blocksize=4000, dtype='int16',
                               channels=1, callback=callback):
            while True:
                data = q.get()
                if recogniser.AcceptWaveform(data):
                    result = recogniser.Result()
                    result_dict = json.loads(result)
                    text = result_dict.get("text", "")
                    words = result_dict.get("result", [])
                    if text:
                        print(f"Recognised: {text}, Confidence: {words}")
                        for phrase in PRESELECTED_PHRASES:
                            phrase_lower = phrase.lower()
                            phrase_words = phrase_lower.split()
                            # Check if the phrase is in the recognised text
                            if phrase_lower == text.lower():
                                # Calculate confidence
                                confidence = calculate_phrase_confidence(phrase_words, words)
                                if confidence is not None:
                                    # Set minimum confidence threshold
                                    MIN_CONFIDENCE = 1
                                    if confidence >= MIN_CONFIDENCE:
                                        print(f"Detected phrase: '{phrase}' with confidence: {confidence:.2f}")
                                else:
                                    print(f"Detected phrase: '{phrase}' but confidence could not be determined.")
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()