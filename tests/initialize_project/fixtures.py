import pytest
from pathlib import Path

@pytest.fixture(scope='session')
def working_dir():
    return Path(__file__).parent

@pytest.fixture(scope="session")
def project_name():
    return 'test_project'
