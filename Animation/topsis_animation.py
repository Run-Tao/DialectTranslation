from manimlib import *

class TOPSIS(Scene):
    def construct(self):
        # Title
        title = Text("TOPSIS: Technique for Order of Preference by Similarity to Ideal Solution").scale(0.4).to_edge(UP)
        self.play(Write(title))

        # Decision Matrix
        matrix_data = [
            ["Alters", "C 1", "C 2", "C 3", "C 4"],
            ["Alt 1", "0.6", "0.3", "0.4", "0.7"],
            ["Alt 2", "0.2", "0.7", "0.6", "0.6"],
            ["Alt 3", "0.8", "0.2", "0.1", "0.4"],
            ["Alt 4", "0.5", "0.5", "0.8", "0.3"]
        ]
        matrix = Matrix(matrix_data).scale(0.3).to_edge(LEFT, buff=1)
        self.play(Write(matrix))

        # Normalize the matrix
        normalized_matrix = matrix.copy().next_to(matrix, DOWN, buff=0.5)
        normalized_matrix[1:].set_color(GREY)
        self.play(Transform(matrix.copy(), normalized_matrix))

        # Highlight the normalized matrix
        normalized_matrix[1:].set_color(WHITE)

        # Weighted Normalized Matrix
        weighted_matrix = normalized_matrix.copy().next_to(normalized_matrix, RIGHT, buff=0.5)
        weighted_matrix[1:].set_color(GREY)
        self.play(Transform(normalized_matrix.copy(), weighted_matrix))

        # Ideal and Anti-Ideal Solutions
        ideal_solution = Tex("Ideal Solution").next_to(weighted_matrix, RIGHT, buff=0.5)
        anti_ideal_solution = Tex("Anti-Ideal Solution").next_to(ideal_solution, RIGHT, buff=0.5)
        self.play(Write(ideal_solution), Write(anti_ideal_solution))

        # Calculate Distances
        distances = Tex("Calculate Distances").next_to(anti_ideal_solution, RIGHT, buff=0.5)
        self.play(Write(distances))

        # Rank the Alternatives
        ranking = Tex("Ranking").next_to(distances, DOWN, buff=0.5)
        self.play(Write(ranking))

        self.wait(2)
