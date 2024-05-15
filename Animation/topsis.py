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
        self.play(ReplacementTransform(Distance,Equation_1_copy))

        TwoDimension = Text("Two Dimension", font="Times New Roman", font_size=16).to_corner(UL, buff = 0.5)

        self.play(Write(TwoDimension))
        self.wait(1)
        MultipleDimensions = Text("Multi-Dimensions", font="Times New Roman", font_size=16).next_to(TwoDimension, DOWN, buff = 0.5)
        MultipleDimensions.shift(DOWN*0.9)

        self.play(Write(MultipleDimensions))

        self.wait(1)

        DownArrow = Tex("\Downarrow").next_to(Equation_1, DOWN)

        self.play(GrowFromPoint(DownArrow, Equation_1.get_center()))
        Equation_2 = Tex("D_n = \sqrt{\sum_{i=1}^n(x_i-y_i)^2}", font_size=28).next_to(DownArrow, DOWN)
        self.play(Write(Equation_2))
        self.wait(1)
        self.play(FadeOut(grid),FadeOut(Dot_1),FadeOut(Dot_2),FadeOut(Arrow_1),FadeOut(Arrow_2))
        self.wait(1)
        self.play(FadeOut(Distance), FadeOut(TwoDimension), FadeOut(DownArrow), FadeOut(Equation_1), FadeOut(Equation_1_copy))
        
        Euclideandistance = Text("Euclidean Distance", font="Times New Roman", font_size=16).to_corner(UL, buff = 0.5)
        Equation_2_copy = Equation_2.copy()
        Equation_2_copy.move_to(Distance)
        self.play(Transform(MultipleDimensions, Euclideandistance),
                  Transform(Equation_2,Equation_2_copy))
        Equation_3 = Tex("S^{+} = \sqrt{\sum_{i=1}^n(v_i-v_i^{+})^2}", font_size=28).next_to(Equation_2, DOWN)

        Equation_4 = Tex("S^{-} = \sqrt{\sum_{i=1}^n(v_i-v_i^{-})^2}", font_size=28).next_to(Equation_3, DOWN)
        explanation_equation_3and4 = Text("TOPSIS Index", font="Times New Roman", font_size=16).next_to(Euclideandistance, DOWN).shift(DOWN*1.5)
        self.play(Write(explanation_equation_3and4),Write(Equation_3),Write(Equation_4))
        self.wait(1)

        SPlus = Tex("S^{+}", font_size=28)

        SMinus = Tex("S^{-}", font_size=28)

        VPlus = Tex("v_i^{+}", font_size=28).next_to(SPlus, DOWN)

        VMinus = Tex("v_i^{-}", font_size=28).next_to(SMinus, DOWN)

        Plus = VGroup(SPlus, VPlus)
        Plus.move_to(Equation_3, RIGHT).shift(RIGHT)

        Minus = VGroup(SMinus, VMinus)
        Minus.move_to(Equation_4, RIGHT).shift(RIGHT)


        self.play(Write(Plus), Write(Minus))

        self.wait(1)

        SPlus_explanation = Text("Similarity to Ideal Solution", font="Times New Roman", font_size=16).next_to(SPlus,RIGHT)
        VPlus_explanation = Text("The coordinates of the Ideal Solution", font="Times New Roman", font_size=16).next_to(VPlus, RIGHT)

        SMinus_explanation = Text("Similarity to Unideal Solution", font="Times New Roman", font_size=16).next_to(SMinus,RIGHT)
        VMinus_explanation = Text("The coordinates of the Unideal Solution", font="Times New Roman", font_size=16).next_to(VMinus, RIGHT)
        
        self.play(Write(SMinus_explanation), Write(SPlus_explanation), Write(VMinus_explanation), Write(VPlus_explanation))
        '''Minus_explanation = VGroup(SMinus_explanation, VMinus_explanation)
        Plus_explanation = VGroup(SPlus_explanation, VPlus_explanation)
        Minus_explanation.move_to(Minus, RIGHT)
        Plus_explanation.move_to(Plus, RIGHT)
        self.play(Write(Plus_explanation), Write(Minus_explanation))'''
        self.wait(1)
        self.play(FadeOut(SMinus_explanation), FadeOut(SPlus_explanation), FadeOut(VMinus_explanation), FadeOut(VPlus_explanation),FadeOut(SMinus),FadeOut(SPlus),FadeOut(VMinus),FadeOut(VPlus),FadeOut(Minus),FadeOut(Plus),FadeOut(explanation_equation_3and4),FadeOut(MultipleDimensions),FadeOut(Equation_2))

        Equation_3_copy = Equation_3.copy().to_corner(UL)
        Equation_4_copy = Equation_4.copy().next_to(Equation_3_copy, DOWN)
        
        self.play(Transform(Equation_3,Equation_3_copy),Transform(Equation_4,Equation_4_copy))
        grid_1 = NumberPlane(color=GREY,axis_config={"include_ticks": False})
        self.play(GrowFromCenter(grid_1))
        # 解释理想解和负理想解
        self.wait(1)

        Dot_plus = Dot(radius=0.08, color=RED_E).move_to(grid_1.coords_to_point(2,3.5))
        Dot_minus = Dot(radius=0.08, color=DARK_BROWN).move_to(grid_1.coords_to_point(-3,-1))

        self.play(Write(Dot_plus), Write(Dot_minus))

        Arrow_plus = Arrow(ORIGIN, Dot_plus, buff=0).set_color(RED)
        Arrow_minus = Arrow(ORIGIN, Dot_minus, buff=0).set_color(GREY_BROWN)

        self.play(GrowArrow(Arrow_plus), GrowArrow(Arrow_minus))

        self.wait(1)
        self.play(FadeOut(Arrow_plus), FadeOut(Arrow_minus))

        Dot_coords = np.random.random((6,2))*10-5
        Dot_m = [Dot(radius=0.05,color=YELLOW).move_to(grid_1.coords_to_point(Dot_coords[i][0],Dot_coords[i][1])) for i in range(len(Dot_coords))]

        Dot_Group = VGroup(Dot_m[0],Dot_m[1],Dot_m[2],Dot_m[3],Dot_m[4],Dot_m[5])

        self.play(Write(Dot_Group))

        self.wait(1)

        self.play(FadeOut(Dot_Group), FadeOut(Dot_plus), FadeOut(Dot_minus), FadeOut(grid_1), FadeOut(Equation_3), FadeOut(Equation_4))

        self.wait(1)

if __name__ == "__main__":
    from os import system
    system("manimgl {} TOPSIS -o -c black".format(__file__))

