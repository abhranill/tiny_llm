
"""Tests for TinyGPT evaluation target counting."""


def count_next_token_targets(token_count):
    """Count next-token targets when the first token is context."""
    if token_count < 2:
        raise ValueError("At least two tokens are required.")

    return token_count - 1


def test_target_count():
    assert count_next_token_targets(663) == 662
    assert count_next_token_targets(2) == 1

    try:
        count_next_token_targets(1)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for one token.")

    print("Target-count tests: PASSED")


if __name__ == "__main__":
    test_target_count()
