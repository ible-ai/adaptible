"""Disjoint structural candidate tests; eligibility never authorizes an adapter."""
import unittest
from adaptible.wrap.question_candidate import candidate_diagnostics

class StructuralCandidateTest(unittest.TestCase):
    def test_structure_without_new_category_vocabulary(self):
        for a,b in (
            ('Which bureau maintains the Silver Acorn Trust?', 'The Silver Acorn Trust is maintained by which bureau?'),
            ('In which district is the Misty Sparrow Studio situated?', 'Which district contains the Misty Sparrow Studio?'),
            ('What is the total charge for the Bronze Kite Pass?', 'How much does it cost to purchase the Bronze Kite Pass?'),
            ('What is the Cloud Rune sculpture carved from?', 'Which material is used to carve the Cloud Rune sculpture?'),
            ('What is the elapsed duration of a Blue Cormorant voyage?', 'How long does a Blue Cormorant voyage last?'),
        ):
            with self.subTest(a=a,b=b):
                self.assertTrue(candidate_diagnostics(a,b)['eligible'])

    def test_boundaries_still_prevent_inversions_aliases_and_extra_clauses(self):
        for a,b in (
            ('Who hires Tessa Reed?', 'Who is hired by Tessa Reed?'),
            ('Who employs Tessa Reed?', 'Who is directly employed by Tessa Reed?'),
            ('Who employs Tessa Reed?', 'Who is employed directly by Tessa Reed?'),
            ('Which bureau maintains the Silver Acorn Trust?', 'Which bureau maintains Silver Acorn?'),
            ('Who directs the Silver Acorn Trust?', 'Who directs the Silver Acorn Trust and the Rose Harbor Fund?'),
            ('What was the charge for the Bronze Kite Pass in 2001?', 'What was the charge for the Bronze Kite Pass in 2011?'),
            ('How much is the Bronze Kite Pass?', 'How much does it cost?'),
            ('Who composed Orchid?', 'The Orchid was composed by whom?'),
            ('Who maintains the Silver Acorn Trust?', 'The Silver Acorn Trust is new but is maintained by whom?'),
            ('Who maintains the Silver Acorn Trust?', 'Who might maintain the Silver Acorn Trust?'),
            ('Who composed "The Orchid"?', 'Orchid was composed by whom?'),
            ('Who maintains the Republic of Evershore Museum?', 'Who maintains the Democratic Republic of Evershore Museum?'),
        ):
            with self.subTest(a=a,b=b):
                self.assertFalse(candidate_diagnostics(a,b)['eligible'])

    def test_unknown_categories_are_only_candidates_not_semantic_acceptance(self):
        a='Which bureau maintains the Silver Acorn Trust?'
        b='Which bureau regulates the Silver Acorn Trust?'
        result=candidate_diagnostics(a,b)
        self.assertTrue(result['eligible'])
        self.assertTrue(result['category_uncertain'])
        self.assertEqual(result['reason'],'candidate_requires_classifier')

if __name__=='__main__':
    unittest.main()
