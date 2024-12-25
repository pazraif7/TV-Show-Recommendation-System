import os
import pickle
import pytest
from unittest.mock import patch
from ShowSuggestAI import generate_embeddings

@pytest.fixture
def setup_test_environment(tmpdir):
    test_csv_path = tmpdir.join("test_tvshows.csv")
    output_folder = tmpdir.mkdir("embeddings_pickle")
    pickle_file_path = os.path.join(output_folder, "embeddings.pkl")
    # Create a mock CSV file
    test_csv_path.write("Title,Description\nGame of Thrones,A fantasy drama.\nBreaking Bad,A chemistry teacher turned meth producer.\n")
    return str(test_csv_path), str(output_folder), str(pickle_file_path)

def test_load_existing_embeddings(setup_test_environment):
    test_csv_path, output_folder, pickle_file_path = setup_test_environment
    # Create a dummy embeddings pickle file
    dummy_embeddings = {"Game of Thrones": [0.1, 0.2, 0.3], "Breaking Bad": [0.4, 0.5, 0.6]}
    with open(pickle_file_path, 'wb') as f:
        pickle.dump(dummy_embeddings, f)
    embeddings = generate_embeddings(tvshows_file=test_csv_path, output_folder=output_folder)
    assert embeddings == dummy_embeddings

def test_generate_embeddings_when_missing(setup_test_environment):
    test_csv_path, output_folder, pickle_file_path = setup_test_environment
    if os.path.exists(pickle_file_path):
        os.remove(pickle_file_path)
    # Mock the OpenAI API to avoid actual API calls
    with patch("openai.Embedding.create") as mock_embedding:
        mock_embedding.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        embeddings = generate_embeddings(tvshows_file=test_csv_path, output_folder=output_folder)
    assert "Game of Thrones" in embeddings
    assert "Breaking Bad" in embeddings
    assert os.path.exists(pickle_file_path)

def test_error_handling_for_missing_api_key(setup_test_environment):
    test_csv_path, output_folder, _ = setup_test_environment
    if "OPENAI_API_KEY" in os.environ:
        del os.environ["OPENAI_API_KEY"]
    with pytest.raises(EnvironmentError, match="OPENAI_API_KEY not found"):
        generate_embeddings(tvshows_file=test_csv_path, output_folder=output_folder)
