import numpy as np
from ShowSuggestAI import find_closest_shows

def test_find_closest_shows_valid_input():
    embeddings = {
        "Game of Thrones": [0.1, 0.2, 0.3],
        "Breaking Bad": [0.4, 0.5, 0.6],
        "Stranger Things": [0.7, 0.8, 0.9],
        "The Witcher": [0.15, 0.25, 0.35],
        "Sherlock": [0.45, 0.55, 0.65]
    }
    average_vector = [0.2, 0.3, 0.4]
    result = find_closest_shows(average_vector, embeddings, top_n=3)
    expected_order = [
        ("The Witcher", 0.9988),
        ("Breaking Bad", 0.9946),
        ("Sherlock", 0.9930)
    ]

    assert len(result) == 3, "The number of closest shows returned is incorrect."
    for (expected_title, expected_similarity), (result_title, result_similarity) in zip(expected_order, result):
        assert expected_title == result_title, f"Expected '{expected_title}' but got '{result_title}'."
        assert np.isclose(expected_similarity, result_similarity, atol=1e-4), \
            f"Similarity for '{result_title}' is incorrect. Expected {expected_similarity}, got {result_similarity}."
