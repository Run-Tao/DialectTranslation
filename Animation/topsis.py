from manimlib import *

class TOPSIS(Scene):
    def construct(self):
        title = Text("Formula Explanation", font="Times New Roman", font_size=32).to_edge(UP)
        self.play(Write(title))
        self.wait(2)

        Gini = Text("Gini Importance", font="Times New Roman", font_size = 20).next_to(title, DOWN)
        Gini.to_edge(LEFT)
        self.play(Write(Gini))

        Gini_equation = Tex("1-\sum_{i=1}^{C_v} p_i^2", font_size = 28).next_to(Gini, RIGHT)

        self.play(Write(Gini_equation))

        self.play(FadeOut(title),FadeOut(Gini), FadeOut(Gini_equation))


        title = Text("TOPSIS",font="Times New Roman", font_size=32).to_edge(UP)
        self.play(Write(title))

        self.wait(2)

        topsis_full_name = Text("Full name: Technique for Order Preference by Similarity to an Ideal Solution", font="Times New Roman", font_size=16).next_to(title,DOWN)

        self.play(Write(topsis_full_name))

        self.wait(2)

        self.play(FadeOut(topsis_full_name))
        self.play(FadeOut(title))
        grid = NumberPlane(color=GREY)
        self.add(grid)
        self.play(GrowFromCenter(grid))
        ''' axes = Axes(
            x_range=[0, 5, 1],  # x轴的范围从0到10，步长为1
            y_range=[0, 5, 1],  # y轴的范围从0到10，步长为1
            axis_config={"include_ticks": True}  # 包含刻度
        )
        
        # 添加坐标轴到场景中
        self.add(axes)
        self.play(FadeIn(axes))
        # 可以选择添加坐标轴标签
        axes_labels = axes.get_axis_labels()
        self.add(axes_labels)
        self.play(FadeIn(axes_labels))
        
        points = [(1,1), (2,3.5), (4,1.2), (3,2.7)]

        scatter = [Dot(radius=0.04, color=BLUE).move_to(axes.coords_to_point(x, y)) for x, y in points]
        
        dots = VGroup(*scatter)

        self.play(FadeIn(dots))

        Arrow_cos = Arrow(start=axes.coords_to_point(0, 0), end=axes.coords_to_point(2, 3.5), buff=0)
        self.play(GrowArrow(Arrow_cos))
        self.wait(1)'''

        Dot_1 = Dot(radius=0.04, color=BLUE).move_to(grid.coords_to_point(2, 3.5))
        Dot_2 = Dot(radius=0.04, color=BLUE).move_to(grid.coords_to_point(3, 2.7))

        self.play(FadeIn(Dot_1), FadeIn(Dot_2))

        Arrow_1 = Arrow(start=grid.coords_to_point(0, 0), end=grid.coords_to_point(2, 3.5), buff=0, color=RED)

        Arrow_2 = Arrow(start=grid.coords_to_point(0,0), end=grid.coords_to_point(3,2.7), buff=0, color=YELLOW)

        Distance = Line(grid.coords_to_point(2,3.5), grid.coords_to_point(3,2.7), color=GREEN)

        self.play(GrowArrow(Arrow_1), GrowArrow(Arrow_2))
        self.play(GrowArrow(Distance))
        self.wait(1)

        Equation_1 = Tex("D_2 = \sqrt{(x_1-x_2)^2+(y_1-y_2)^2}", font_size=28).to_corner(UL, buff=0.5)
        
        self.play(Transform(Distance,Equation_1))

        Equation_1_copy = Equation_1.copy().shift(RIGHT*3)
        Equation_1.move_to(Equation_1_copy)
        self.play(ReplacementTransform(Distance,Equation_1_copy),)
        self.wait(1)

        TwoDimension = Text("Two Dimension", font="Times New Roman", font_size=16).to_corner(UL, buff = 0.5)

        self.play(Write(TwoDimension))
        self.wait(1)
        MultipleDimensions = Text("Multi-Dimensions", font="Times New Roman", font_size=16).next_to(TwoDimension, DOWN, buff = 0.5)
        MultipleDimensions.shift(DOWN*0.9)

        self.play(Write(MultipleDimensions))

        self.wait(1)



        DownArrow = Tex("\Downarrow").next_to(Equation_1, DOWN)

        self.play(GrowFromPoint(DownArrow, Equation_1.get_center()))
        Equation_2 = Tex("D_n = \sqrt{\sum_{i=1}^n(x_i-x_j)^2+(y_i-y_j)^2}", font_size=28).next_to(DownArrow, DOWN)
        self.play(Write(Equation_2))
        self.wait(1)

if __name__ == "__main__":
    from os import system
    system("manimgl {} TOPSIS -o -c black".format(__file__))

