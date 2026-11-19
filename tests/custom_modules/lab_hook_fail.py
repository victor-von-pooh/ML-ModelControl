from typing import Callable

from lab_units import ScaleBlock


def register_units(register: Callable) -> None:
    """
    1 つ登録してから誤りを出す関数 (誤りの包み方を確かめるため)

    Parameters
    ----------
    register: Callable
        register_unit と同じ引数を受け取る登録の関数

    Returns
    ----------
    None
    """
    register(ScaleBlock, name="hookfail.Scale")
    raise RuntimeError("部品の準備に失敗")
