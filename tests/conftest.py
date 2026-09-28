import pytest

LIBRARY_A = [
    "CCO",
    "CCN",
    "CCC",
    "c1ccccc1",
    "c1ccccc1O",
    "c1ccccc1N",
    "CC(=O)O",
    "CC(=O)Nc1ccc(O)cc1",
    "CCOc1ccc2nc(S(N)(=O)=O)sc2c1",
    "CN1CCC[C@H]1c1cccnc1",
    "OC(=O)c1ccccc1O",
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "O=C(O)CCc1ccccc1",
    "NCCc1ccc(O)c(O)c1",
    "CC(C)NCC(O)c1ccc(O)c(O)c1",
    "C1CCOC1",
    "ClCCl",
    "CC(=O)OC1=CC=CC=C1C(=O)O",
    "COc1ccc(CCN)cc1",
    "Cc1ccccc1",
]

LIBRARY_B = LIBRARY_A[::-1]


@pytest.fixture
def library_a():
    return list(LIBRARY_A)


@pytest.fixture
def library_b():
    return list(LIBRARY_B)
