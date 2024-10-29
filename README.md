## 概要

『ポケモンコロシアム』のリライブ作業を自動化します。

## 使い方

メニューのカーソルを「ポケモン」に合わせておき、ダークポケモン研究所にあるパソコンの前に立って実行ボタンを押してください。

## 詳細

マクロには2つのモードがあり、それぞれ以下の手順でリライブ作業を行います。

***手持ちポケモンのみリライブ***

1. 研究所の周りを10周し、リライブゲージを確認
2. 手持ち全てのリライブゲージが白くなったら、アゲトビレッジへ移動しリライブを実行

***ボックスのポケモンもまとめてリライブ***

1. 研究所の周りを10周し、リライブゲージを確認
2. リライブゲージが白くなっているポケモンがいた場合、アゲトビレッジに移動してリライブを実行
3. 研究所に戻り、リライブ済みのポケモンとボックス内のダークポケモンを入れ替えて作業を続行

リライブ実行時の進化やレベルアップ技は、全てキャンセル処理を行っています。

## 注意事項

- 欧州版は60hzのみの対応となります。

## 動作環境

- Poke-Controller Modified
- Poke-Controller Modified Extension

## Requirements

- [NumPy](https://github.com/numpy/numpy)
- [OpenCV-Python](https://github.com/opencv/opencv-python)
