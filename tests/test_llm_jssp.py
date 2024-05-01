import pytest


@pytest.mark.skipif({"Pigs"}.issubset({"Fly"}), reason="no way of currently testing this")
@pytest.mark.null_feature
def test_basic_1():
    assert True, "no tests here!"

@pytest.mark.skip(reason="no way of currently testing this")
def test_basic_2():
    assert True, "no tests here!"

if __name__ == "__main__":
    pytest.main()
