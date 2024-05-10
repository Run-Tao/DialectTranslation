from manimlib import *
import numpy as np

class DialectTranslation(Scene):
    def construct(self):

        title = Text("Dialect Translation", font="Times New Roman",font_size=32).to_edge(UP)
        Purpose = Text("Purpose: The research aims at turning Chinese dialects into mandarin", font="Times New Roman", font_size = 16).next_to(title, DOWN)
        BasicTheory = Text("Basic Theory: Finding a map between Chinese dialect and mandarin", font="Times New Roman", font_size = 16).next_to(Purpose, DOWN)

        self.play(Write(title))
        self.wait(2)
        self.play(Write(Purpose))
        self.wait(2)
        self.play(Write(BasicTheory))
        self.wait(2)
        self.play(FadeOut(title), FadeOut(Purpose), FadeOut(BasicTheory))
        
        title = Text("Basic Theory", font="Times New Roman",font_size=32).to_edge(UL)

        self.play(Write(title))
        self.wait(2)
        left_axes = Axes(
            height=4,
            width=4,
            x_range=(-2, 2, 0.5),
            y_range=(-2, 2, 0.5)
        ).to_edge(LEFT)

        # 创建右侧坐标轴
        right_axes = Axes(
            height=4,
            width=4,
            x_range=(-2, 2, 0.5),
            y_range=(-2, 2, 0.5)
        ).to_edge(RIGHT)

        # 生成在[-1, 1]范围内的随机点
        points = np.random.random((100, 2)) * 2 - 1
        # 创建散点并指定点的大小为0.04
        scatter = [Dot(radius=0.04, color=BLUE).move_to(left_axes.coords_to_point(x, y)) for x, y in points]
        points = np.random.random((100, 2)) * 2 - 1
        # 将散点添加到左侧坐标轴上
        left_dots = VGroup(*scatter)
        self.add(left_axes, left_dots)

        # 为右侧坐标轴创建一个相同的点集
        right_scatter = [Dot(radius=0.04, color=YELLOW).move_to(right_axes.coords_to_point(x, y)) for x, y in points]
        right_dots = VGroup(*right_scatter)
        self.add(right_axes, right_dots)

        self.wait(2)

        group = VGroup(left_axes, left_dots)
        group_copy = group.copy()
        origingroup = group.copy()
        origingroup.move_to(ORIGIN)

        # 创建移动动画
        

        group_2 = VGroup(right_axes, right_dots)
        group_2_copy = group_2.copy()
        origingroup_2 = group_2.copy()
        origingroup_2.move_to(ORIGIN)

        # 创建移动动画
        self.play(Transform(group_copy,origingroup))
        self.play(Transform(group_2_copy, origingroup_2))
        self.wait(2)

if __name__ == "__main__":
    from os import system
    system("manimgl {} DialectTranslation -o -c black".format(__file__))