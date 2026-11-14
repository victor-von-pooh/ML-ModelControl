import ml_modelcontrol


def test_version() -> None:
    """
    パッケージを読み込んでバージョンを取得できることを確かめる関数

    Parameters
    ----------
    None

    Returns
    ----------
    None
    """
    # 導入したパッケージのバージョンは文字列
    assert isinstance(ml_modelcontrol.__version__, str)
