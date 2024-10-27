from Commands.Keys import Button, Hat, Direction
from Commands.PythonCommandBase import ImageProcPythonCommand
from time import perf_counter, sleep
from collections import deque
from enum import Enum, auto
from typing import List, Optional
from datetime import timedelta
import numpy as np
import os
import cv2
import tempfile
import time

from . import image_process

direction_to_hat = {
    (-1, 0): Hat.TOP,
    (1, 0): Hat.BTM,
    (0, -1): Hat.LEFT,
    (0, 1): Hat.RIGHT
}

def directions_to_hats(directions):
    return [direction_to_hat[dir] for dir in directions]
    
def print_pokemon_box(box):
    cols = len(box[0])
    output = []
    output.append("+" + "---" * cols + "+")
    for row in box:
        display_row = "|"
        for cell in row:
            if cell == Status.Shadow:
                display_row += " D "
            elif cell == Status.Normal:
                display_row += " N "
            elif cell == Status.Recovering:
                display_row += " R "
            elif cell is None:
                display_row += " - "
        display_row += "|"
        output.append(display_row)
    output.append("+" + "---" * cols + "+")
    print("\n".join(output))

def loading_message(wait_time, message):
    dots = ""
    count = 3
    interval = wait_time / count

    print(message, end="")
    for _ in range(count):
        sleep(interval)
        dots += "."
        print(".", end="")
    print()
    
class Status(Enum):
    Normal = auto()
    Shadow = auto()
    Recovering = auto()

class CoPurificationAutomate(ImageProcPythonCommand):
    NAME = "Co_リライブ自動化"

    def __init__(self, cam):
        super().__init__(cam)
        self.icon_crop_area = [
            # (142, 135, 81, 64),
            (142, 339, 81, 64),
            (611, 82, 80, 64),
            (611, 202, 80, 64),
            (611, 323, 80, 64),
            (611, 443, 80, 64)
        ]
        self.party_crop_area = [
            (129, 125, 441, 144),
            (129, 329, 441, 144),
            (598, 73, 576, 84),
            (598, 190, 576, 84),
            (598, 314, 576, 84),
            (598, 434, 576, 84),
        ]
        self.box_crop_area = (485, 174, 708, 480)
        self.gauge_crop_area = (820, 491, 81, 17)
        self.select_crop_area = (172, 311, 217, 104)
        self.partner_crop_area = (580, 420, 100, 150)
        self.party_list = None
        self.box_list = None
        self.box_count = 3
        self.cur_box_num = 1
        self.partner = None
        self.start_time = time.time()
        
    def do(self):
        option_list = ["手持ちポケモンだけリライブ", "ボックスのポケモンもまとめてリライブ"]
        option = self.dialogue6widget("Option",[["Radio", "選択", option_list, option_list[0]]])
        if not option:
            return
        self.open_party_menu()
        self.party_list = self.detect_shadow_in_party()
        self.close_party_menu()
        if option[0] == option_list[1]:
            self.access_storage_system()
            self.detect_box_state()
            self.swap_party_and_box_pokemon(Status.Shadow)
            self.close_storage_system()
            print(f"残りのダークポケモン: {self.all_count_status(Status.Shadow)}匹")
        if not self.contains_status(Status.Shadow):
            print("ダークポケモンが存在しないためマクロを終了します")
            self.finish()
        self.exit_laboratory()
        if option[0] == option_list[0]:
            self.party_only_purification()
        else:
            self.all_purification()
        txt = "リライブ作業が完了しました"
        print(txt)
        self.LINE_text(txt)
        
    def all_purification(self):
        """ボックスと手持ちのダークポケモンをリライブ"""
        
        while True:
            self.circling_around_labo(10)
            self.open_party_menu()
            self.recovering_check()
            self.close_party_menu()
            if Status.Recovering in self.party_list:
                self.goto_agate_village()
                self.enter_relic_cave()
                self.enter_reric_forest()
                for party_idx, status in enumerate(self.party_list):
                    if status == Status.Recovering:
                        self.exec_purification(party_idx)
                print(f"残りのダークポケモン: {self.all_count_status(Status.Shadow)}匹")
                if not self.contains_status(Status.Shadow):
                    break
                self.exit_relic_forest()
                self.goto_laboratory()
                self.enter_laboratory()
                if self.count_status_in_box(Status.Shadow) > 0:
                    self.access_storage_system()
                    self.swap_party_and_box_pokemon(Status.Shadow)
                    self.close_storage_system()
                self.exit_laboratory()
            
    def party_only_purification(self):
        """手持ちのダークポケモンだけリライブ"""
        
        while True:
            self.circling_around_labo(10)
            self.open_party_menu()
            self.recovering_check()
            self.close_party_menu()
            filtered_list = [status for status in self.party_list if status in (Status.Shadow, Status.Recovering)]
            if all(status == Status.Recovering for status in filtered_list):
                break
        self.goto_agate_village()
        self.enter_relic_cave()
        self.enter_reric_forest()
        for party_idx, status in enumerate(self.party_list):
            if status == Status.Recovering:
                self.exec_purification(party_idx)
    
    def detect_box_state(self):
        """
        ボックスの状態を検出する
        ボックス名にカーソルを合わせておく
        """
        # ボックスが全てのポケモンを読み込むまで少し待つ
        # ?マークが表示されていると上手く検出できない
        loading_message(2.5, "ボックス内のポケモンを検出中")
        box_list = []
        for i in range(self.box_count):
            if i != 0:
                self.switching_next_box(Button.R)
                self.wait(0.5) # OBS経由による遅延を考慮
            box_list.append(self.detect_shadow_in_box(0.03))
        for count, box in enumerate(box_list, start=1):
            print(f"Box {count}:")
            print_pokemon_box(box)
        self.box_list = box_list
    
    def swap_party_and_box_pokemon(self, target_status):
        """
        手持ちのポケモンをボックスのポケモンと入れ替える
        ボックス名にカーソルを合わせておくこと
        """
        self.press(Hat.BTM)
        cur_cell = (0, 3)
        grab_pokemon = None
        for i, status in enumerate(self.party_list):
            box_num = self.find_status_in_box(target_status)
            if box_num is None:
                break
            if status != target_status:
                self.switching_jump_box(box_num)
                cur_cell, grab_pokemon = self.swap_pokemon(box_num, cur_cell, i, grab_pokemon, target_status)
                self.party_list[i] = target_status
        
        # 手持ちが6匹未満の場合
        for _ in range(6 - len(self.party_list)):
            box_num = self.find_status_in_box(target_status)
            if box_num is None:
                break
            self.switching_jump_box(box_num)
            cur_cell, grab_pokemon = self.swap_pokemon(box_num, cur_cell, len(self.party_list), grab_pokemon, target_status)
            self.party_list.append(target_status)
        
        # 手持ちがいっぱいで連れていけない場合
        if grab_pokemon is not None:
            box_num = self.find_status_in_box(None)
            self.switching_jump_box(box_num)
            directions, (row, col) = self.search_nearest_status(self.box_list[box_num - 1], cur_cell[0], cur_cell[1], None)
            for hat in directions_to_hats(directions):
                self.press(hat)
            self.press(Button.A, wait=0.5)  # ボックスの空きにポケモンを置く
            self.box_list[box_num - 1][row][col] = grab_pokemon

    def swap_pokemon(self, box_num, cur_cell, party_index, grab_pokemon, target_status):
        """
        ボックス内のダークポケモンと手持ちのポケモンを入れ替える
        """
        directions, cell = self.search_nearest_status(self.box_list[box_num - 1], cur_cell[0], cur_cell[1], target_status)
        if cell == (-1, -1):
            return cur_cell, grab_pokemon
        row, col = cell
        for hat in directions_to_hats(directions):
            self.press(hat)
        self.press(Button.A, wait=0.5)  # ボックス内のダークポケモンを掴む
        self.box_list[box_num - 1][row][col] = grab_pokemon
        self.pressRep(Hat.TOP, row + 2)
        self.press(Button.A, wait=0.8)  # ボックス上で手持ちを開く
        self.pressRep(Hat.BTM, party_index)
        self.press(Button.A, wait=0.5)  # 手持ちと入れ替える or 空きにセットする
        grab_pokemon = self.party_list[party_index] if party_index < len(self.party_list) else None
        self.press(Button.B, wait=0.8)  # 手持ちを閉じる
        return (0, 0), grab_pokemon

    def exit_laboratory(self):
        """
        研究所の外に出る
        PCの前に立っているのが前提
        """
        self.press(Direction.RIGHT, duration=1.3, wait=0.02)
        self.hold(Direction.DOWN)
        if self.wait_load(5):
            self.holdEnd(Direction.DOWN)
            self.wait_until_load_finishes()
            self.press(Direction.DOWN, duration=3.3)
        else:
            self.stop_macro("Failed to exit laboratory.")
    
    def enter_laboratory(self):
        """
        研究所に入ってPCの前に立つまで
        """
        self.hold(Direction.UP, wait=2)
        if self.wait_load(10):
            self.holdEnd(Direction.UP)
            self.wait_until_load_finishes()
            self.press(Direction.UP_LEFT, duration=0.5, wait=0.02)
            self.press(Direction.LEFT, duration=0.8, wait=0.02)
            self.press(Direction.UP, duration=0.2)
        else:
            self.stop_macro("Failed to enter laboratory.")

    def circling_around_labo(self, loop_count: int):
        """
        研究所の周りをぐるぐる走り回る
        考案: 奈都さん(@Natu5307051)
        """
        self.press(Direction.RIGHT, duration=4, wait=0.02)
        for i in range(loop_count):
            self.press(Direction.UP, duration=6.6, wait=0.02)
            self.press(Direction.LEFT, duration=7.8, wait=0.02)
            self.press(Direction.DOWN, duration=6.6, wait=0.02)
            if i == loop_count - 1:
                self.press(Direction.RIGHT, duration=3.8, wait=0.5)
            else:
                self.press(Direction.RIGHT, duration=7.8, wait=0.02)
                
    def goto_laboratory(self):
        """洞窟入口 -> 研究所"""
        
        wait_time = 0.05
        self.press(Direction.DOWN_LEFT, duration=0.9, wait=wait_time)
        self.press(Direction.DOWN_RIGHT, duration=1.8, wait=wait_time)
        self.press(Direction.RIGHT, duration=1.8, wait=wait_time)
        self.press(Direction.DOWN, duration=0.55, wait=wait_time)
        self.press(Direction.LEFT, duration=1, wait=wait_time)
        self.press(Direction.DOWN, duration=0.7, wait=wait_time)
        self.press(Direction.LEFT, duration=2.5, wait=wait_time)
        self.press(Direction.UP, duration=1.7, wait=wait_time)
        self.press(Direction.LEFT, duration=1.2, wait=wait_time)
        self.hold(Direction.DOWN, 2)
        if self.wait_load(10):
            self.holdEnd(Direction.DOWN)
            self.wait_until_load_finishes()
            self.press(Direction.DOWN_RIGHT, wait=0.7)
            self.press(Button.A)
        else:
            self.stop_macro("Failed to goto laboratory.")
            
        if self.wait_load(10):
            self.wait_until_load_finishes()
    
    def goto_agate_village(self):
        self.hold(Direction.DOWN)
        if self.wait_load(10):
            self.holdEnd(Direction.DOWN)
            self.wait_until_load_finishes()
            self.press(Direction.UP_LEFT, wait=0.7)
            self.press(Button.A)
        else:
            self.stop_macro("Failed to goto agate village.")
            
        if self.wait_load(10):
            self.wait_until_load_finishes()
        
    def enter_relic_cave(self, in_center = False):
        """ポケセン->洞窟, アゲトビレッジ入口->洞窟"""
        
        wait_time=0.05
        if not in_center:
            self.press(Direction.UP, duration=2.5, wait=wait_time)
            self.press(Direction.RIGHT, duration=1.4, wait=wait_time)
            self.press(Direction.DOWN, duration=1.6, wait=wait_time)
            self.press(Direction.RIGHT, duration=2.4, wait=wait_time)
            self.press(Direction.UP, duration=0.6, wait=wait_time)
            self.press(Direction.RIGHT, duration=0.8, wait=wait_time)
            self.press(Direction.UP, duration=0.55, wait=wait_time)
            self.press(Direction.LEFT, duration=1.7, wait=wait_time)
        else:
            self.press(Direction.DOWN_LEFT, duration=1, wait=0.02)
            self.press(Direction.LEFT, duration=0.3, wait=0.02)
            self.hold(Direction.DOWN)
            if self.wait_load(10):
                self.holdEnd(Direction.DOWN)
                self.wait_until_load_finishes()
            self.press(Direction.LEFT, duration=2, wait=wait_time)
            
        self.press(Direction.UP_LEFT, duration=1, wait=wait_time)
        self.press(Direction.LEFT, duration=0.3, wait=wait_time)
        self.hold(Direction.UP)
        if self.wait_load(5):
            self.holdEnd(Direction.UP)
            self.wait_until_load_finishes()
        else:
            self.stop_macro("Failed to enter relic cave.")
    
    def enter_reric_forest(self):
        """洞窟入口 ～ 石の前まで"""
        
        self.press(Direction.UP, duration=0.7, wait=0.02)
        self.press(Direction.UP_RIGHT, duration=2.25, wait=0.02)
        self.press(Direction.UP, duration=1, wait=0.02)
        self.press(Direction.UP_RIGHT, duration=1.8, wait=0.02)
        self.hold(Direction.UP, wait=1)
        if self.wait_load(5, threshold=0.95):
            self.holdEnd(Direction.UP)
            self.wait_until_load_finishes()
            self.press(Direction.UP, duration=2.2, wait=0.02)
        else:
            self.stop_macro("Failed to enter relic forest.")
    
    def exit_relic_forest(self):
        """石の前 -> 洞窟入口"""
        
        self.press(Direction.LEFT, duration=0.15, wait=0.02)
        self.hold(Direction.DOWN, wait=1)
        # 洞窟に入る
        if self.wait_load(10):
            self.holdEnd(Direction.DOWN)
            self.wait_until_load_finishes()
        else:
            self.stop_macro("Failed to exit relic forest.")
        self.press(Direction.DOWN, duration=0.5, wait=0.02)
        self.press(Direction.DOWN_LEFT, duration=1.6, wait=0.02)
        self.press(Direction.DOWN, duration=0.8, wait=0.02)
        self.press(Direction.DOWN_LEFT, duration=2.5, wait=0.02)
        self.hold(Direction.DOWN)
        # 洞窟から出る
        if self.wait_load(5, threshold=0.95):
            self.holdEnd(Direction.DOWN)
            self.wait_until_load_finishes()
    
    def enter_pokemon_center(self):
        """洞窟入口 -> ポケモンセンター"""
        
        self.press(Direction.DOWN_LEFT, duration=1.0, wait=0.02)
        self.press(Direction.DOWN_RIGHT, duration=1.8, wait=0.02)
        self.press(Direction.RIGHT, duration=2, wait=0.02)
        self.hold(Direction.UP)
        # ポケセンに入る
        if self.wait_load(10):
            self.holdEnd(Direction.UP)
            self.wait_until_load_finishes()
        self.press(Direction.UP, duration=0.8, wait=0.02)
        self.press(Direction.RIGHT, duration=0.5, wait=0.02)
        self.press(Direction.UP_RIGHT, duration=0.5, wait=0.02)
    
    def exec_purification(self, party_index):
        """
        リライブを実行する
        石の前に立っているのが前提
        """
        
        self.wait(0.5)
        img = self.camera.readFrame()
        
        if self.partner is None:
            frame = image_process.crop_image(img, *self.partner_crop_area)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
                self.partner = temp.name
                cv2.imwrite(self.partner, frame)
        
        prev_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        while True:
            next_img = cv2.cvtColor(self.camera.readFrame(), cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(prev_img, next_img)
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            ratio = np.sum(thresh  > 0) / (thresh.shape[0] * thresh.shape[1])
            self._logger.debug(ratio)
            if ratio > 0.5:
                break
            self.press(Button.A, wait=0.5)
        
        self.wait(0.7)
        self.pressRep(Hat.BTM, party_index)
        self.press(Button.A)
        if not self.wait_load(10):
            self.stop_macro("Failed transition to purification scene.")
        self.wait_until_load_finishes()
        self.wait(8)
        lower = np.array([0, 0, 0])
        upper = np.array([180, 255, 50])
        prev_img = cv2.cvtColor(image_process.crop_image(self.camera.readFrame(), *self.select_crop_area), cv2.COLOR_BGR2GRAY)
        while True:
            img = self.camera.readFrame()
            if image_process.calc_color_ratio(img, lower, upper) > 0.9:
                break
            next_img = cv2.cvtColor(image_process.crop_image(img, *self.select_crop_area), cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(prev_img, next_img)
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            ratio = np.sum(thresh  > 0) / (thresh.shape[0] * thresh.shape[1])
            self._logger.debug(ratio)
            if ratio > 0.3:
                self.press(Button.B, wait=0.6)
                self.pressRep(Button.A, 2)
            self.press(Button.B)
        self.wait_until_load_finishes()
        
        while not self.isContainTemplate(self.partner, 0.8):
            self.press(Button.B)
        self.party_list[party_index] = Status.Normal
        self.wait(0.3)
        
    def open_party_menu(self):
        self.press(Button.X, wait=0.5)
        self.press(Button.A, wait=1.8)
    
    def close_party_menu(self):
        self.pressRep(Button.B, 2, wait=1.2)
        self.press(Button.B, wait=0.5)
        
    def recovering_check(self):
        self.pressRep(Button.A, 2)
        if self.wait_load(5):
            self.wait_until_load_finishes()
            self.wait(0.5)
        for i, status in enumerate(self.party_list):
            if i != 0:
                self.press(Hat.BTM, wait=0.6)
            if status == Status.Shadow:
                ratio = self.calc_heart_gauge_ratio()
                self._logger.debug(ratio)
                if ratio == 0:
                    self.party_list[i] = Status.Recovering
        self.press(Button.B, wait=0.8)
        
    def access_storage_system(self):
        self.cur_box_num = 1
        
        self.press(Button.A)
        if self.wait_load(5):
            self.wait_until_load_finishes()
            self.press(Button.A, wait=2)
            self.press(Hat.TOP)
            self.press(Button.Y, wait=0.3)
        else:
            self.stop_macro("Failed to access storage system.")
    
    def close_storage_system(self):
        self.pressRep(Button.B, 2, wait=1)
        self.press(Button.B)
        if self.wait_load(5):
            self.wait_until_load_finishes()
        else:
            self.stop_macro("Failed to close storage system.")
    
    def switching_next_box(self, direction: Button):
        if direction == Button.R:
            self.press(direction,  wait=0.65)
            self.cur_box_num += 1
            if self.cur_box_num > 3:
                self.cur_box_num = 1
        elif direction == Button.L:
            self.press(direction,  wait=0.65)
            self.cur_box_num -= 1
            if self.cur_box_num < 1:
                self.cur_box_num = 3
    
    def switching_jump_box(self, box_num: int):
        if self.cur_box_num == box_num:
            return
        
        right_distance = (box_num - self.cur_box_num + self.box_count) % self.box_count
        left_distance = (self.cur_box_num - box_num + self.box_count) % self.box_count
        if right_distance <= left_distance:
            for _ in range(right_distance):
                self.press(Button.R, wait=0.65)
        else:
            for _ in range(left_distance):
                self.press(Button.L, wait=0.65)
        
        self.cur_box_num = box_num

    def search_nearest_status(self, box, start_row, start_col, target_status):
        rows, cols = len(box), len(box[0])
        if box[start_row][start_col] == target_status:
            return [], (start_row, start_col)

        queue = deque([(start_row, start_col, [])])
        visited = [[False] * cols for _ in range(rows)]
        visited[start_row][start_col] = True
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while queue:
            row, col, path = queue.popleft()
            for dr, dc in directions:
                r, c = row + dr, col + dc
                if 0 <= r < rows and 0 <= c < cols and not visited[r][c]:
                    visited[r][c] = True
                    new_path = path + [(dr, dc)]
                    if box[r][c] == target_status:
                        return new_path, (r, c)
                    queue.append((r, c, new_path))
        return [], (-1, -1)
    
    def detect_party_pokemon_count(self, threshold=0.8):
        """
        現在の手持ちポケモンの数を検出する
        """
        
        img = self.camera.readFrame()
        party_count = 1
        lower = np.array([0, 0, 0])
        upper = np.array([180, 255, 50])
        for (x, y, w, h) in self.icon_crop_area:
            crop_img = img[y:y+h, x:x+w]
            ratio = image_process.calc_color_ratio(crop_img, lower, upper)
            self._logger.debug(ratio)
            if ratio < threshold:
                party_count += 1
            else:
                break
        return party_count
    
    def detect_shadow_in_party(self, threshold=0.3):
        """
        手持ちにダークポケモンが居るか調べる
        
        """
        loading_message(1.0, "手持ちのポケモンを検出中")
        img = self.camera.readFrame()
        party_list = []
        party_count = self.detect_party_pokemon_count()
        lower = np.array([35, 40, 40])
        upper = np.array([85, 255, 255])

        for i in range(party_count):
            (x, y, w, h) = self.party_crop_area[i]
            crop_img = img[y:y+h, x:x+w]
            ratio = image_process.calc_color_ratio(crop_img, lower, upper)
            self._logger.debug(f"Pokemon {i+1}: ratio = {ratio}")
            if ratio > threshold:
                party_list.append(Status.Normal)
            else:
                party_list.append(Status.Shadow)
        result = " | ".join([f"{i+1}: {'N' if status == Status.Normal else 'D' if status == Status.Shadow else '-'}" for i, status in enumerate(party_list)])
        print(result)
        return party_list
        
    def detect_shadow_in_box(self, threshold=0.03) ->  List[List[Optional[Status]]]:
        img1 = image_process.crop_image(self.camera.readFrame(), *self.box_crop_area)
        self.wait(0.25)
        img2 = image_process.crop_image(self.camera.readFrame(), *self.box_crop_area)
        self.wait(0.25)
        img3 = image_process.crop_image(self.camera.readFrame(), *self.box_crop_area)

        rows, cols = 5, 6
        height = img1.shape[0] // rows
        width = img1.shape[1] // cols

        lower = np.array([120, 50, 50])
        upper = np.array([170, 255, 255])
        
        poke_box = self.detect_pokemon_in_box(img1)
        result = [] 

        for i in range(rows):
            row_result = [] 
            for j in range(cols):
                if poke_box[i][j]:
                    y1 = i * height
                    x1 = j * width
                    y2 = y1 + height
                    x2 = x1 + width
                    
                    crop_img1 = img1[y1:y2, x1:x2]
                    crop_img2 = img2[y1:y2, x1:x2]
                    crop_img3 = img3[y1:y2, x1:x2]
                    
                    diff1 = cv2.absdiff(crop_img1, crop_img2)
                    diff2 = cv2.absdiff(crop_img2, crop_img3)
                    diff3 = cv2.absdiff(crop_img1, crop_img3)

                    hsv_diff1 = cv2.cvtColor(diff1, cv2.COLOR_BGR2HSV)
                    hsv_diff2 = cv2.cvtColor(diff2, cv2.COLOR_BGR2HSV)
                    hsv_diff3 = cv2.cvtColor(diff3, cv2.COLOR_BGR2HSV)

                    mask1 = cv2.inRange(hsv_diff1, lower, upper)
                    mask2 = cv2.inRange(hsv_diff2, lower, upper)
                    mask3 = cv2.inRange(hsv_diff3, lower, upper)

                    ratio1 = np.count_nonzero(mask1) / mask1.size
                    ratio2 = np.count_nonzero(mask2) / mask2.size
                    ratio3 = np.count_nonzero(mask3) / mask3.size
                    is_shadow = (ratio1 > threshold) or (ratio2 > threshold) or (ratio3 > threshold)
                    if is_shadow:
                        row_result.append(Status.Shadow)
                        self._logger.debug(f"Shadow detected. Cell ({i}, {j}) Ratios: {ratio1}, {ratio2}, {ratio3})")
                    else:
                        row_result.append(Status.Normal)
                        self._logger.debug(f"Normal detected. Cell ({i}, {j}) Ratios: {ratio1}, {ratio2}, {ratio3})")
                else:
                    row_result.append(None)
            result.append(row_result) 
        return result

    def detect_pokemon_in_box(self, img, lower_threshold=30, upper_threshold=100, threshold=0.01):
        rows, cols = 5, 6
        height = img.shape[0]  // rows
        width = img.shape[1] // cols
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_img = cv2.GaussianBlur(gray_img, (5, 5), 1.5)
        result = []
        
        for i in range(rows):
            row_result = [] 
            for j in range(cols):
                grid = blur_img[i*height:(i+1)*height, j*width:(j+1)*width]
                edges = cv2.Canny(grid, lower_threshold, upper_threshold)
                ratio = np.sum(edges > 0) / edges.size
                self._logger.debug(ratio)
                row_result.append(ratio >= threshold)
            result.append(row_result)
        return result
    
    def find_status_in_box(self, status):
        if self.box_list is None:
            return None
        for i, box in enumerate(self.box_list, start=1):
            for row in box:
                if any(pokemon == status for pokemon in row):
                    return i
        return None
    
    def count_status_in_party(self, status):
        count = 0
        if self.box_list:
            for box in self.box_list:
                for row in box:
                    count += sum(1 for pokemon in row if pokemon == status)
        return count
    
    def count_status_in_box(self, status):
        if self.party_list:
            return sum(1 for pokemon in self.party_list if pokemon == status)
        return 0

    def all_count_status(self, status):
        """指定した状態を持つポケモンが何匹居るか調べる"""
        
        return self.count_status_in_party(status) + self.count_status_in_box(status)

    def contains_status(self, status):
        """指定した状態を持つポケモンが存在するか調べる"""
        
        return self.all_count_status(status) > 0
            
    def wait_load(self, timeout, threshold=0.9):
        """
        暗転するまで待つ
        """
        
        start = perf_counter()
        lower = np.array([0, 0, 0])
        upper = np.array([180, 255, 50])
        while True:
            ratio = image_process.calc_color_ratio(self.camera.readFrame(), lower, upper) 
            self._logger.debug(ratio)
            if ratio > threshold:
                return True
            elapsed_time = perf_counter() - start
            if elapsed_time > timeout:
                return False
            self.checkIfAlive()
            
    def wait_until_load_finishes(self, threshold=0.9):
        """
        暗転が終わる（ロードが終わる）まで待つ
        """
        
        lower = np.array([0, 0, 0])
        upper = np.array([180, 255, 50])
        while image_process.calc_color_ratio(self.camera.readFrame(), lower, upper) > threshold:
            self.checkIfAlive()
    
    def press_while_loading(self, button, duration=0.1, wait=0.1, threshold=0.9):
        """
        暗転するまでボタンを押す
        """
        
        lower = np.array([0, 0, 0])
        upper = np.array([180, 255, 50])
        while not image_process.calc_color_ratio(self.camera.readFrame(), lower, upper) > threshold:
            self.press(button, duration, wait)
        
    def calc_heart_gauge_ratio(self):
        img = image_process.crop_image(self.camera.readFrame(), *self.gauge_crop_area)
        lower = np.array([125, 100, 50])
        upper = np.array([135, 255, 255])
        return image_process.calc_color_ratio(img, lower, upper)

    def is_contain_template_wait(self, template, wait):
        current_time = perf_counter()    
        while not self.isContainTemplate(template, 0.8):
            if perf_counter() - current_time > wait :
                return False
        return True
    
    def stop_macro(self, error_message: str):
        self._logger.debug(error_message)
        print(f"問題が発生したためマクロを停止しました: {error_message}")
        self.finish()
    
    def end(self, ser):
        print(f"-- Execution time: {timedelta(seconds=time.time() - self.start_time)} -- ")
        if self.partner is not None:
            os.remove(self.partner)
        super().end(ser)