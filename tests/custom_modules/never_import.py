# 定義に書いた名前からは読み込まれないことを確かめるモジュール.
# 読み込まれると印を立てる
import builtins

builtins.ML_MODELCONTROL_NEVER_IMPORTED = True
