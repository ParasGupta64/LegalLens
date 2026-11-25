"""Streamlit smoke test with inference isolated from network and model weights."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'article_src' / 'src'))
from multilingual_judgebert import generate_explanation, load_checkpoint
from streamlit.testing.v1 import AppTest


class AppTests(unittest.TestCase):
    def test_evaluator_and_chat_navigation(self):
        prediction = dict(lmp_score=3., decision='Not Preserved', context_used=True,
                          error_types=['obligation_shift'],
                          error_probabilities={'obligation_shift': .8},
                          explanation='Test diagnostic.')
        with patch('multilingual_judgebert.load_checkpoint', return_value=(None, None, 'context_aware')), \
             patch('multilingual_judgebert.predict_single_pair', return_value=prediction):
            app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
            self.assertFalse(app.exception)
            app.sidebar.radio[0].set_value('💬  AI Document Chat').run()
            self.assertFalse(app.exception)

    def test_explanations_do_not_assert_legal_equivalence(self):
        text = generate_explanation(10., 'Preserved', ['none'])
        self.assertIn('model predicts', text)
        self.assertIn('does not verify legal equivalence', text)

    def test_checkpoint_pointer_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'model.pt').write_text('version https://git-lfs.github.com/spec/v1\n')
            with self.assertRaisesRegex(ValueError, 'git lfs pull'):
                load_checkpoint(directory)


if __name__ == '__main__':
    unittest.main()
