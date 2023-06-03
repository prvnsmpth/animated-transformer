import numpy as np

from manim import *

class TwoColumnMapping(Scene):

    def __init__(self, text, vectors, xlabel = None, ylabel = None):
        self.text = text
        self.vectors = vectors
        self.xlabel = xlabel
        self.ylabel = ylabel
        super().__init__()

    def construct(self):
        self.show_embeddings() 
    
    def show_embeddings(self):
        words = self.text.split(' ')

        grid = VGroup()
        word_group = VGroup()
        vector_group = VGroup()
        arrow_group = VGroup()

        for (i, word) in enumerate(words):
            word_obj = Text(f'"{word}"')
            arrow_obj = Arrow(start=LEFT, end=RIGHT)
            embedding = [ Tex(x) for x in self.vectors[i] ]
            if len(self.vectors[i]) > 1:
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

        if self.xlabel:
            word_brace = Brace(grid, direction=LEFT).set_color(GREEN)
            word_label = word_brace.get_text(self.xlabel).rotate(np.pi / 2).set_color(GREEN)

        if self.ylabel:
            dim_brace = Brace(vector_group, direction=UP).set_color(BLUE)
            dim_label = dim_brace.get_text(self.ylabel).set_color(BLUE)

        self.add(word_group)
        if self.xlabel:
            self.play(FadeIn(word_brace, word_label))
        self.play(FadeIn(arrow_group))
        if self.ylabel:
            self.play(FadeIn(vector_group, dim_brace, dim_label))
        else:
            self.play(FadeIn(vector_group))

        self.wait(4)


class TransformerFunc(Scene):

    def __init__(self, show_tokenization=False):
        super().__init__()
        self.show_tokenization = show_tokenization

    def construct(self):

        title1 = Text("The GPT function").move_to(UP * 2).scale(0.6)
        title2 = Text("Given word sequence, return next word").move_to(UP * 2).scale(0.6)
        title3 = Text("...with tokenization").move_to(UP * 2).scale(0.6)

        p1 = Tex("$GPT($")
        p2 = Tex("$X$").set_color(BLUE)
        p3 = Tex("$) \\rightarrow$")
        p4 = Tex("$Y$").set_color(GREEN)
        eq_group = VGroup(p1, p2, p3, p4)
        eq_group.arrange()

        p2_words = Tex("$``the \\; robots\\;  are\\;  coming\"$").shift(LEFT).set_color(BLUE)
        p4_word = Tex("$``tonight\"$").next_to(p2_words, RIGHT * 5).set_color(GREEN)

        p2_ids = Tex("$[39, 12, 104, 108]$").set_color(BLUE)
        p4_id = Tex("$899$").next_to(p2_ids, RIGHT * 5).set_color(GREEN)

        self.play(Write(title1), FadeIn(eq_group))
        self.wait(1)
        eq_group = VGroup(p1, p2_words, p3, p4_word)
        self.play(
            FadeOut(title1), Write(title2),
            FadeOut(p2, p4), FadeIn(p2_words, p4_word), eq_group.animate.arrange())
        self.wait(2)

        if self.show_tokenization:
            eq_group = VGroup(p1, p2_ids, p3, p4_id)
            self.play(
                FadeOut(title2), Write(title3),
                FadeOut(p2_words, p4_word), FadeIn(p2_ids, p4_id), eq_group.animate.arrange())

        self.wait(4)



class PositionEmbeddings(TwoColumnMapping):

    def __init__(self):
        super().__init__("0 1 2 3", [
            [0.60, 0.49, 0.79],
            [0.26, 0.65, 0.78],
            [0.10, 0.80, 0.63],
            [0.93, 0.71, 0.64]
        ], xlabel="T = (up to) 1024", ylabel="C = 786")


class WordEmbeddings(TwoColumnMapping):

    def __init__(self):
        super().__init__("39(the) 12(robots) 104(are) 108(coming)", [
            [0.31, 0.48, 0.41],
            [0.35, 0.24, 0.69],
            [0.51, 0.02, 0.63],
            [0.63, 0.77, 0.03]
        ], xlabel="T = (up to) 1024", ylabel="C = 786")

class Tokenization(TwoColumnMapping):

    def __init__(self):
        super().__init__("aardvark apple box cardboard", [
            [13],
            [42],
            [314],
            [271]
        ], xlabel="All words", ylabel="Unique ID")


class SelfAttention(Scene):

    def construct(self):
        self.camera.background_color = WHITE
        self.show_token_embed()

    def show_token_embed(self):
        input = [[0.31, 0.48, 0.41],
            [0.35, 0.24, 0.69],
            [0.51, 0.02, 0.63],
            [0.63, 0.77, 0.03]]
        M = len(input)
        N = len(input[0])

        tr_input = []
        for row in input:
            tr_row = [ Tex(e) for e in row ]
            # tr_row.insert(-1, Tex("\\dots"))
            tr_input.append(tr_row)
        # tr_input.append([ Tex("\\vdots") for _ in range(N + 1) ])

        input_matrix = MobjectMatrix(tr_input)
        word_brace = Brace(input_matrix, direction=LEFT).set_color(GREEN)
        word_label = word_brace.get_text("T = 4").rotate(np.pi / 2).set_color(GREEN)
        dim_brace = Brace(input_matrix, direction=UP).set_color(BLUE)
        dim_label = dim_brace.get_text("C = 786").set_color(BLUE)

        token_embed = VGroup(input_matrix, word_brace, word_label, dim_brace, dim_label)
        token_embed.set_color(BLACK)

        self.play(
            Write(Text("Token embedding matrix").scale(0.7).shift(UP * 3.5), run_time=1.0),
            FadeIn(token_embed.scale(0.9).shift(DOWN)),
        )
        self.add(
            Text("T = Number of tokens in the input").next_to(token_embed, DOWN).scale(0.6),
            Text("C = Size of embedding for each token").next_to(token_embed, DOWN * 3).scale(0.6))

        boxes = []
        arrows = []
        tex_words = []
        tokens = "the robots are coming".split(' ')
        for (row, token) in zip(input_matrix.get_rows(), tokens):
            box = SurroundingRectangle(row)
            boxes.append(box)

            right = box.get_right()
            arrow_start = right + UP + 2 * RIGHT
            arrow = Arrow(arrow_start, right, color=GOLD)
            arrows.append(arrow)

            tex_word = Tex(f'``{token}"').set_color(YELLOW).move_to(arrow_start + UP * 0.2)
            tex_words.append(tex_word)
        
        self.play(Create(boxes[0]), Create(arrows[0]), Write(tex_words[0]))
        for i in range(1, len(boxes)):
            self.play(
                ReplacementTransform(boxes[i - 1], boxes[i]), 
                ReplacementTransform(arrows[i - 1], arrows[i]),
                ReplacementTransform(tex_words[i - 1], tex_words[i]))
        
        self.wait(2)
        l = len(boxes) - 1
        self.remove(boxes[l], arrows[l], tex_words[l])

        



