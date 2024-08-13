"""Tests for metrics"""
import pytest

import numpy as np

from utils.cosine_similarity import CosineSimilarity

@pytest.fixture
def cosine_similarity():
    return CosineSimilarity()

def test_cosine_similarity(cosine_similarity):
    np.random.seed(0)
    # When the two vectors are the same, the cosine similarity should be 1
    x = np.array([1, 2, 3])
    y = np.array([2, 4, 6])
    result = cosine_similarity.compute(predictions=[x], references=[y])
    assert result['cosine_similarity'] == pytest.approx(1)

    # When the two vectors are orthogonal, the cosine similarity should be 0
    x = np.array([1, 0, 1, 0])
    y = np.array([0, 1, 0, 1])
    result = cosine_similarity.compute(predictions=[x], references=[y])
    assert result['cosine_similarity'] == pytest.approx(0)

    # When the two vectors are opposite, the cosine similarity should be -1
    x = np.array([1, 2, 3])
    y = np.array([-1, -2, -3])
    result = cosine_similarity.compute(predictions=[x], references=[y])
    assert result['cosine_similarity'] == pytest.approx(-1)

def test_cosine_similarity_2d_matrix(cosine_similarity):
    np.random.seed(0)
    # Test with 2D arrays
    x = np.array([[1, 2, 3], [2, 0, 0], [4, 4, 4]])
    y = np.array([[-2, -4, -6], [0, 1, 0], [2, 2, 2]])
    result = cosine_similarity.compute(predictions=x, references=y)
    assert result['cosine_similarity'] == pytest.approx(0)  # Average of -1, 0, and 1