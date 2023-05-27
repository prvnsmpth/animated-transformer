import numpy as np

from manim import *

class EmbeddingsScene(Scene):

    def __init__(self, text, vectors):
        self.text = text
        self.vectors = vectors
        super().__init__()

    def construct(self):
        self.show_embeddings() 
    
    def show_embeddings(self):
        words = self.text.split(' ')
        num_words = len(words)

        grid = VGroup()
        word_group = VGroup()
        vector_group = VGroup()
        arrow_group = VGroup()

        for (i, word) in enumerate(words):
            word_obj = Text(f'"{word}"')
            arrow_obj = Arrow(start=LEFT, end=RIGHT)
            embedding = [ Tex(x) for x in self.vectors[i] ]
            embedding.insert(-1, Tex("\\dots"))

            vector_obj = VGroup(*list(embedding))
            vector_obj.arrange()

            grid += word_obj
            grid += arrow_obj
            grid += vector_obj

            word_group += word_obj
            vector_group += vector_obj
            arrow_group += arrow_obj

        word_dots = Tex("\\vdots")
        grid += word_dots
        word_group += word_dots
        dot_arrow = Arrow(start=LEFT, end=RIGHT)
        arrow_group += dot_arrow
        grid += dot_arrow
        vector_dots = Tex("\\vdots")
        vector_group += vector_dots
        grid += vector_dots

        grid.arrange_in_grid(5, 3, buff=0.4)

        word_brace = Brace(grid, direction=LEFT).set_color(GREEN)
        word_label = word_brace.get_text("T = 1024").rotate(np.pi / 2).set_color(GREEN)

        dim_brace = Brace(vector_group, direction=UP).set_color(BLUE)
        dim_label = dim_brace.get_text("C = 786").set_color(BLUE)

        self.add(word_group)
        self.play(FadeIn(word_brace, word_label))
        self.play(FadeIn(arrow_group))
        self.play(FadeIn(vector_group, dim_brace, dim_label))


class PositionEmbeddings(EmbeddingsScene):

    def __init__(self):
        super().__init__("0 1 2 3", [
            [0.60, 0.49, 0.79],
            [0.26, 0.65, 0.78],
            [0.10, 0.80, 0.63],
            [0.93, 0.71, 0.64]
        ])


class WordEmbeddings(EmbeddingsScene):

    def __init__(self):
        super().__init__("the robots are coming", [
            [0.31, 0.48, 0.41],
            [0.35, 0.24, 0.69],
            [0.51, 0.02, 0.63],
            [0.63, 0.77, 0.03]
        ])


class SelfAttention(Scene):

    def construct(self):
        
        input = [[0.63, 0.62, 0.89, 0.93],
           [0.85, 0.99, 0.55, 0.97],
           [0.86, 0.50, 0.68, 0.39],
           [0.37, 0.08, 0.11, 0.63]]
        M = len(input)
        N = len(input[0])

        tr_input = []
        for row in input:
            tr_row = [ Tex(e) for e in row ]
            tr_row.insert(-1, Tex("\\dots"))
            tr_input.append(tr_row)
        tr_input.append([ Tex("\\vdots") for _ in range(N + 1) ])

        input_matrix = MobjectMatrix(tr_input)
        word_brace = Brace(input_matrix, direction=LEFT).set_color(GREEN)
        word_label = word_brace.get_text("T = 1024").rotate(np.pi / 2).set_color(GREEN)
        dim_brace = Brace(input_matrix, direction=UP).set_color(BLUE)
        dim_label = dim_brace.get_text("C = 786").set_color(BLUE)

        token_embed = VGroup(input_matrix, word_brace, word_label, dim_brace, dim_label)

        self.add(token_embed)

        self.play(token_embed.animate.scale(0.5))
        self.play(token_embed.animate.shift(LEFT * 5))

        



