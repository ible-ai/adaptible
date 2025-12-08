"""Test loop detection functionality."""

from adaptible._src._llm import _detect_token_loop


def test_no_loop_short_sequence():
    """Test that no loop is detected when sequence is too short."""
    tokens = [1, 2, 3, 4, 5]
    assert not _detect_token_loop(tokens, sequence_length=3, max_repetitions=3)


def test_no_loop_different_sequences():
    """Test that no loop is detected when sequences are different."""
    tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert not _detect_token_loop(tokens, sequence_length=3, max_repetitions=3)


def test_loop_detected_exact():
    """Test that loop is detected when exact repetition occurs."""
    # Sequence [1, 2, 3] repeated 3 times
    tokens = [1, 2, 3, 1, 2, 3, 1, 2, 3]
    assert _detect_token_loop(tokens, sequence_length=3, max_repetitions=3)


def test_loop_detected_longer_sequence():
    """Test loop detection with longer sequences."""
    # Sequence [10, 20, 30, 40] repeated 3 times
    tokens = [10, 20, 30, 40, 10, 20, 30, 40, 10, 20, 30, 40]
    assert _detect_token_loop(tokens, sequence_length=4, max_repetitions=3)


def test_loop_not_detected_partial_repetition():
    """Test that partial repetition doesn't trigger loop detection."""
    # Sequence [1, 2, 3] repeated only 2 times (not enough)
    tokens = [1, 2, 3, 1, 2, 3]
    assert not _detect_token_loop(tokens, sequence_length=3, max_repetitions=3)


def test_loop_detected_with_prefix():
    """Test loop detection when there's a prefix before the loop."""
    # Prefix [99, 98] then [1, 2, 3] repeated 3 times
    tokens = [99, 98, 1, 2, 3, 1, 2, 3, 1, 2, 3]
    assert _detect_token_loop(tokens, sequence_length=3, max_repetitions=3)


def test_loop_not_detected_broken_pattern():
    """Test that a broken pattern doesn't trigger loop detection."""
    # Pattern breaks on the third repetition
    tokens = [1, 2, 3, 1, 2, 3, 1, 2, 4]
    assert not _detect_token_loop(tokens, sequence_length=3, max_repetitions=3)


def test_loop_detected_single_token():
    """Test loop detection with single token repetition."""
    # Token [5] repeated 5 times
    tokens = [5, 5, 5, 5, 5]
    assert _detect_token_loop(tokens, sequence_length=1, max_repetitions=5)


def test_loop_edge_case_exact_length():
    """Test when token list is exactly the minimum length needed."""
    # Exactly 9 tokens: [1, 2, 3] repeated 3 times
    tokens = [1, 2, 3, 1, 2, 3, 1, 2, 3]
    assert _detect_token_loop(tokens, sequence_length=3, max_repetitions=3)


if __name__ == "__main__":
    # Run all tests
    test_no_loop_short_sequence()
    print("✓ test_no_loop_short_sequence")

    test_no_loop_different_sequences()
    print("✓ test_no_loop_different_sequences")

    test_loop_detected_exact()
    print("✓ test_loop_detected_exact")

    test_loop_detected_longer_sequence()
    print("✓ test_loop_detected_longer_sequence")

    test_loop_not_detected_partial_repetition()
    print("✓ test_loop_not_detected_partial_repetition")

    test_loop_detected_with_prefix()
    print("✓ test_loop_detected_with_prefix")

    test_loop_not_detected_broken_pattern()
    print("✓ test_loop_not_detected_broken_pattern")

    test_loop_detected_single_token()
    print("✓ test_loop_detected_single_token")

    test_loop_edge_case_exact_length()
    print("✓ test_loop_edge_case_exact_length")

    print("\n✅ All tests passed!")
