"""Watson NLP emotion detection client."""

import json
import requests

URL = (
    "https://sn-watson-emotion.labs.skills.network/v1/"
    "watson.runtime.nlp.v1/NlpService/EmotionPredict"
)
HEADERS = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}


def emotion_detector(text_to_analyze):
    """Return emotion scores and dominant emotion for the supplied text."""
    response = requests.post(
        URL,
        headers=HEADERS,
        json={"raw_document": {"text": text_to_analyze}},
        timeout=30,
    )
    if response.status_code == 400:
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None,
        }
    response.raise_for_status()
    data = json.loads(response.text)
    scores = data["emotionPredictions"][0]["emotion"]
    emotions = {name: scores[name] for name in (
        "anger", "disgust", "fear", "joy", "sadness"
    )}
    emotions["dominant_emotion"] = max(emotions, key=emotions.get)
    return emotions
