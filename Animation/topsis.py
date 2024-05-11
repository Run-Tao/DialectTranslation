from manimlib import *

class TOPSIS(Scene):
    def construct(self):
        title = Text("TOPSIS",font="Times New Roman", font_size=32).to_edge(UP)
        self.play(Write(title))

        self.wait(2)

        topsis_full_name = Text("Full name: Technique for Order Preference by Similarity to an Ideal Solution", font="Times New Roman", font_size=16).next_to(title,DOWN)

        self.play(Write(topsis_full_name))

        self.wait(2)

        self.play(FadeOut(topsis_full_name))
        self.play(FadeOut(title))
        grid = NumberPlane(axis_config={"stroke_color": WHITE, "stroke_width": 0, "include_ticks": False})
        self.add()
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

if __name__ == "__main__":
    from os import system
    system("manimgl {} TOPSIS -o -c black".format(__file__))

