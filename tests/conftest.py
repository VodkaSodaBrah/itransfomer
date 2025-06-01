import pytest
import sys
import os

# Add the project root to the path so imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture(scope='session')
def setup_mps():
    # Setup code for MPS if needed
    pass

@pytest.fixture
def sample_data():
    # Provide sample data for testing
    return {
        'input': ...,
        'expected_output': ...
    }