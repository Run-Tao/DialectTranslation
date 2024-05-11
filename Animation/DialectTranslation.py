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
        points = np.random.random((50, 2)) * 2 - 1
        # 创建散点并指定点的大小为0.04
        scatter = [Dot(radius=0.04, color=BLUE).move_to(left_axes.coords_to_point(x, y)) for x, y in points]
        points = np.random.random((50, 2)) * 2 - 1
        # 将散点添加到左侧坐标轴上
        left_dots = VGroup(*scatter)
        self.add(left_axes, left_dots)

        # 为右侧坐标轴创建一个相同的点集
        right_scatter = [Dot(radius=0.04, color=YELLOW).move_to(right_axes.coords_to_point(x, y)) for x, y in points]
        right_dots = VGroup(*right_scatter)
        self.play(FadeIn(right_axes), FadeIn(right_dots))

        Axe_name_left = Tex("\mathcal{M}, stands for \mathbf{M}andarin").next_to(left_axes, UP)
        Axe_name_right = Tex("\mathcal{D}, stands for \mathbf{D}ialect").next_to(right_axes, UP)
        self.play(Write(Axe_name_left), Write(Axe_name_right))
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
        group_blueyellow = VGroup(origingroup,origingroup_2)
        group_blueyellow_copy = VGroup(origingroup, origingroup_2).scale(1.5)
        # 创建移动动画
        self.play(FadeOut(Axe_name_left), FadeOut(Axe_name_right), FadeOut(group), FadeOut(group_2), Transform(group_copy,origingroup), Transform(group_2_copy, origingroup_2))
        self.play(ReplacementTransform(group_blueyellow,group_blueyellow_copy))
        self.wait(2)
        # 若有空，做一个蓝色点平移到黄色点上的视频，以体现映射效果

        self.play(*[FadeOut(mob) for mob in self.mobjects])

        title = Text("Research Methods", font="Times New Roman",font_size=32).to_edge(UL)
        self.play(Write(title))
        self.wait(2)

        method_SVD = Text("SVD", font_size=26, font="Times New Roman").move_to(ORIGIN)
        method_MFCCs = Text("MFCCs", font="Times New Roman", font_size=26).next_to(method_SVD, UP, buff =0.5)
        
        method_Map = Text("Map", font_size=26, font="Times New Roman").next_to(method_SVD, DOWN, buff =0.5)

        self.play(Write(method_MFCCs), Write(method_SVD), Write(method_Map))
        self.wait(2)
        method_SVD_copy = method_SVD.to_edge(LEFT)
        self.play(Transform(method_SVD, method_SVD_copy), Transform(method_MFCCs, method_MFCCs.next_to(method_SVD, UP, buff=0.5)), Transform(method_Map, method_Map.next_to(method_SVD, DOWN, buff=0.5)))
        self.wait(2)

        MFCCS_full = Text("Full name: Multi-Frequency Cepstral Coefficients", font="Times New Roman", font_size=16).next_to(method_MFCCs, RIGHT)
        SVD_full = Text("Full name: Singular Value Decomposition", font="Times New Roman", font_size=16).next_to(method_SVD, RIGHT)
        MAP_explanation = Text("Using Cosine Distance and Linear Regression to find a map for dialects", font="Times New Roman", font_size=16).next_to(method_Map, RIGHT)
        # 遗留问题 动画没做好
        self.play(Write(MFCCS_full), Write(SVD_full), Write(MAP_explanation))
        self.wait(2)

if __name__ == "__main__":
    from os import system
    system("manimgl {} DialectTranslation -o -c black".format(__file__))