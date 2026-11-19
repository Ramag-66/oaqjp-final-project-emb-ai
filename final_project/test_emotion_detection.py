"""Unit tests for emotion detection with deterministic mocked Watson responses."""
import json
import unittest
from unittest.mock import patch
from EmotionDetection import emotion_detector


class TestEmotionDetection(unittest.TestCase):
    """Verify emotion parsing and selection without network dependency."""

    def check_dominant(self, expected):
        """Check that the highest score determines the dominant emotion."""
        scores = {key: 0.05 for key in (
            "anger", "disgust", "fear", "joy", "sadness"
        )}
        scores[expected] = 0.8
        payload = {"emotionPredictions": [{"emotion": scores}]}
        with patch("EmotionDetection.emotion_detection.requests.post") as post:
            post.return_value.status_code = 200
            post.return_value.text = json.dumps(payload)
            result = emotion_detector("Sample text")
        self.assertEqual(result["dominant_emotion"], expected)

    def test_joy(self):
        """Recognize joy as the dominant emotion."""
        self.check_dominant("joy")

    def test_anger(self):
        """Recognize anger as the dominant emotion."""
        self.check_dominant("anger")

    def test_disgust(self):
        """Recognize disgust as the dominant emotion."""
        self.check_dominant("disgust")

    def test_sadness(self):
        """Recognize sadness as the dominant emotion."""
        self.check_dominant("sadness")

    def test_fear(self):
        """Recognize fear as the dominant emotion."""
        self.check_dominant("fear")


if __name__ == "__main__":
    unittest.main()
