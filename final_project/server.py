"""Flask server for the Emotion Detector application."""
from flask import Flask, render_template, request
from EmotionDetection import emotion_detector

app = Flask(__name__)


@app.route("/")
def index():
    """Render the emotion detector user interface."""
    return render_template("index.html")


@app.route("/emotionDetector")
def detect_emotion():
    """Analyze user text and return a readable result."""
    text_to_analyze = request.args.get("textToAnalyze", "")
    if not text_to_analyze.strip():
        return "Invalid text! Please try again!"
    try:
        result = emotion_detector(text_to_analyze)
    except (ValueError, KeyError, IndexError, OSError) as exc:
        return f"Emotion detection failed: {exc}", 502
    if result["dominant_emotion"] is None:
        return "Invalid text! Please try again!"
    return (
        "For the given statement, the system response is "
        f"'anger': {result['anger']}, 'disgust': {result['disgust']}, "
        f"'fear': {result['fear']}, 'joy': {result['joy']} "
        f"and 'sadness': {result['sadness']}. "
        f"The dominant emotion is {result['dominant_emotion']}."
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
