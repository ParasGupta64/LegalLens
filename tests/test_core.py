"""Regression checks without network calls or model downloads."""
import io
import sys
import unittest
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'article_src' / 'src'))
from doc_chat import (build_vector_index, chat_with_document, chunk_text,
                      retrieve_relevant_chunks, validate_uploaded_file)
from prepare_multilingual_training_corpora import (create_stratified_splits,
                                                   create_unrelated_augmentation)
from ui_formatting import _escape_html, _md_to_html


class ChatTests(unittest.TestCase):
    def test_untrusted_html_is_escaped_before_formatting(self):
        output = _md_to_html('**Hello**\n> <img src=x onerror=alert(1)>\n<script>bad</script>')
        self.assertIn('<strong>Hello</strong>', output)
        self.assertIn('<blockquote', output)
        self.assertNotIn('<img', output)
        self.assertNotIn('<script', output)
        self.assertIn('&lt;img', output)
        self.assertEqual(_escape_html('<b>\n&'), '&lt;b&gt;<br>&amp;')

    def test_chunking_covers_text_without_repeating_the_final_tail(self):
        text = 'Clause text. ' * 150
        chunks = chunk_text(text)
        self.assertEqual(chunks[-1]['end'], len(text.strip()))
        self.assertEqual(sum(c['end'] == len(text.strip()) for c in chunks), 1)
        self.assertEqual(chunk_text('short'), [dict(chunk_id=0, text='short', start=0, end=5)])
        for first, second in zip(chunks, chunks[1:]):
            self.assertGreater(second['start'], first['start'])
            self.assertLessEqual(second['start'], first['end'])
        for size, overlap in [(0, 0), (-1, 0), (10, -1), (10, 10)]:
            with self.assertRaises(ValueError):
                chunk_text('text', size, overlap)

    def test_retrieval_abstains_on_no_match_or_missing_index(self):
        chunks = chunk_text('The tenant must pay rent monthly.')
        index = build_vector_index(chunks)
        self.assertIsNotNone(index)
        self.assertEqual(retrieve_relevant_chunks('astronomy', index, chunks), [])
        self.assertEqual(retrieve_relevant_chunks('rent', None, chunks), [])
        self.assertEqual(retrieve_relevant_chunks('rent', index, chunks, top_k=0), [])
        self.assertEqual(len(retrieve_relevant_chunks('rent', index, chunks)), 1)
        answer, generated = chat_with_document('astronomy', chunks, index, [])
        self.assertFalse(generated)
        self.assertIn('does not appear', answer)
        self.assertIsNone(build_vector_index([dict(text='! ?')]))

    def test_upload_validation_rejects_empty_large_and_unreadable_files(self):
        stream = io.BytesIO(b'pdf data')
        stream.seek(3)
        self.assertEqual(validate_uploaded_file(stream, 'document.PDF'), (True, ''))
        self.assertEqual(stream.tell(), 3)
        self.assertFalse(validate_uploaded_file(io.BytesIO(), 'empty.pdf')[0])
        self.assertFalse(validate_uploaded_file(io.BytesIO(b'x' * (20 * 1024 * 1024 + 1)), 'large.pdf')[0])
        self.assertFalse(validate_uploaded_file(object(), 'unreadable.pdf')[0])
        self.assertFalse(validate_uploaded_file(stream, 'legacy.doc')[0])


class SplitTests(unittest.TestCase):
    def test_source_clauses_never_cross_partitions(self):
        records = [dict(pair_id=str((lang, i, j)), language=lang, domain='contract',
                        original_text=f'{lang} original {i}', simplified_text=f'candidate {j}')
                   for lang in ('en', 'hi') for i in range(12) for j in range(3)]
        # A repeated original in another bucket must still stay in its source group.
        records.append(dict(records[0], pair_id='duplicate', domain='statute'))
        splits = create_stratified_splits(records)
        self.assertEqual(splits, create_stratified_splits(records))
        self.assertEqual(sum(map(len, splits)), len(records))
        self.assertTrue(all(splits))
        sources = [{r['original_text'] for r in split} for split in splits]
        for left, right in combinations(sources, 2):
            self.assertFalse(left & right)

    def test_unrelated_augmentation_handles_a_single_source(self):
        record = dict(pair_id='one', language='en', domain='contract',
                      context_text='', original_text='Only source')
        self.assertEqual(create_unrelated_augmentation([record]), [])
        self.assertEqual(create_unrelated_augmentation([record, dict(record, pair_id='two')]), [])


if __name__ == '__main__':
    unittest.main()
