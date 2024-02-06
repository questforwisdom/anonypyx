import numpy as np
from anonypyx import ksame

import pytest

@pytest.fixture
def prepared_images():
    images = [
        [
            [32, 32, 32],
            [64, 64, 64],
            [128, 128, 128],
            [0, 0, 0]
        ],
        [
            [16, 15, 16],
            [63, 64, 65],
            [127, 120, 129],
            [0, 1, 2]
        ],
        [
            [100, 50, 200],
            [100, 50, 200],
            [100, 50, 200],
            [100, 50, 200]
        ],
        [
            [101, 45, 201],
            [99, 51, 200],
            [100, 47, 189],
            [103, 49, 192]
        ]
    ]

    return np.array(images)

def test_kSamePixel(prepared_images):
    anonymizer = ksame.kSame(prepared_images, 3, 4, k=2, variant='pixel', clustering_implementation='Random Choice')

    result, mapping = anonymizer.anonymize()
    expected1 = np.mean(np.array([prepared_images[0], prepared_images[1]]), axis=0)
    expected2 = np.mean(np.array([prepared_images[2], prepared_images[3]]), axis=0)

    assert (expected1 == result[mapping[0]]).all()
    assert (expected1 == result[mapping[1]]).all()
    assert (expected2 == result[mapping[2]]).all()
    assert (expected2 == result[mapping[3]]).all()
