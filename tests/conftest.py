import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="session")
def app_module():
    """Import the real app.py once per test session.

    This loads the actual committed data files (data/interactions_processed.csv,
    data/recipes_improved.csv, data/recipes_vectors.pkl) exactly as the deployed
    app does, so tests exercise the real module-level wiring rather than a mock.
    Individual tests then call parseReviews/generateRecommendations with their
    own small synthetic data instead of the loaded production data.
    """
    import app

    return app
