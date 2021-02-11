# MFRL
卒業研究で強化学習の学習時間を短縮する手法を提案．

強化学習で使うシミュレータ内の数値計算の近似精度を学習中に動的に変更することで学習時間を短縮する．実験では4次のルンゲクッタ法が使われているシミュレータを用いて，学習の始めは1次のルンゲクッタ法を使い，徐々に2次3次4次と近似精度を上げていくことで最終的な学習結果は悪化させずに学習時間を短縮している．

## 使用したライブラリ
Python：3.7.6

Pytorch：1.3.1

NumPy：1.18.1

SciPy：1.4.1

## 各フォルダ
code：実験に使用したPythonコード．

result：実験結果のtxtファイルを使って実験結果を表示するPythonコード．

acrobot.py：ルンゲクッタの近似次数を変更できるようにOpenAI Gymからダウンロードしたacrobot.pyを改変したもの．実行する際にはダウンロードしたOpenAI Gymのacrobot.pyをこのacrobot.pyに差し替える．
