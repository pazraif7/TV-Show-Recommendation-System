from ShowSuggestAI import match_shows

def test_match_shows():
    csv_path = "imdb_tvshows - imdb_tvshows.csv"
    user_input = ["gem of throns", "lupan", "witcher"]
    expected_output = ["Game of Thrones", "Lupin", "The Witcher"]
    assert match_shows(user_input, csv_path) == expected_output
