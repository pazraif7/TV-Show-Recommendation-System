import pytest
import numpy as np
from ShowSuggestAI import average_vector

@pytest.fixture
def setup_embeddings():
    return {
        "Game of Thrones": [0.1, 0.2, 0.3],
        "Breaking Bad": [0.4, 0.5, 0.6],
        "Stranger Things": [0.7, 0.8, 0.9]
    }

def test_average_vector_valid_input(setup_embeddings):
    embeddings = setup_embeddings
    matched_shows = ["Game of Thrones", "Breaking Bad"]
    avg_vector = average_vector(matched_shows, embeddings)
    expected_vector = np.mean([embeddings["Game of Thrones"], embeddings["Breaking Bad"]], axis=0)
    assert np.allclose(avg_vector, expected_vector), "Average vector does not match expected value."
