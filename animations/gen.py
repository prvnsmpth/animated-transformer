import numpy as np

from manim import *

VOCAB = {
    'the': 3206,
    'robots': 2736,
    'will': 3657,
    'bring': 400,
    'prosperity': 2532,
    'aardvark': 14,
    'apple': 177,
    'box': 392,
    'cardboard': 477
}

WORD_EMB = {
    'the': [0.07213349, 0.13476127, 0.63486506],
    'robots': [0.8144936 , 0.51136212, 0.43797063],
    'will': [0.61763848, 0.29474857, 0.80120351],
    'bring': [0.5001875 , 0.15730688, 0.93133526],
    'prosperity': [0.2479659 , 0.41104862, 0.23145515]
}

POS_EMB = {
    '0': [0.3204288 , 0.11540252, 0.88234465],
    '1': [0.14030786, 0.07085044, 0.40712213],
    '2': [0.07433667, 0.47350997, 0.22633834],
    '3': [0.80512473, 0.1189765 , 0.0632117 ]
}

class TwoColumnMapping(Scene):

    def __init__(self, text, embeddings, xlabel = None, ylabel = None, quote_tokens=True, show_dot_row=False):
        self.text = text
        self.vectors = [embeddings[t] for t in text.split(' ')]
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.quote_tokens = quote_tokens
        self.show_dot_row = show_dot_row
        super().__init__()

        self.camera.background_color = WHITE
        Text.set_default(color=BLACK)
        Tex.set_default(color=BLACK)
        Arrow.set_default(color=BLACK)
        BraceLabel.set_default(color=BLACK)

    def construct(self):
        self.show_embeddings() 
    
    def show_embeddings(self):
        words = self.text.split(' ')

        grid = VGroup()
        word_group = VGroup()
        vector_group = VGroup()
        arrow_group = VGroup()

        for (i, word) in enumerate(words):
            word_obj = Text(f'"{word}"') if self.quote_tokens else Text(word)
            arrow_obj = Arrow(start=LEFT, end=RIGHT)
            word_value = self.vectors[i]
            if not isinstance(word_value, list):
                word_value = [word_value]
            embedding = [ Tex(f'{x:.2f}') if type(x) == 'float' else Tex(x) for x in word_value ]
            if len(embedding) > 1:
                embedding.insert(-1, Tex("\\dots"))

            vector_obj = VGroup(*list(embedding))
            vector_obj.arrange()

            grid += word_obj
            grid += arrow_obj
            grid += vector_obj

            word_group += word_obj
            vector_group += vector_obj
            arrow_group += arrow_obj

        if self.show_dot_row:
            word_dots = Tex("\\vdots")
            grid += word_dots
            word_group += word_dots
            dot_arrow = Arrow(start=LEFT, end=RIGHT)
            arrow_group += dot_arrow
            grid += dot_arrow
            vector_dots = Tex("\\vdots")
            vector_group += vector_dots
            grid += vector_dots

        grid.arrange_in_grid(len(words) if not self.show_dot_row else len(words) + 1, 3, buff=0.4)

        if self.xlabel:
            word_brace = BraceLabel(grid, self.xlabel, brace_direction=LEFT).set_color(GREEN_E)
            word_brace.label.rotate(np.pi / 2).set_color(GREEN_E).shift(RIGHT * 0.5)

        if self.ylabel:
            dim_brace = BraceLabel(vector_group, self.ylabel, brace_direction=UP).set_color(BLUE_E)
            # dim_label = dim_brace.get_text(self.ylabel).set_color(BLUE)

        self.add(word_group)
        if self.xlabel:
            self.play(FadeIn(word_brace))
        self.play(FadeIn(arrow_group))
        if self.ylabel:
            self.play(FadeIn(vector_group, dim_brace))
        else:
            self.play(FadeIn(vector_group))

        self.wait(4)


class TransformerFunc(Scene):

    def __init__(self, show_tokenization=True):
        super().__init__()
        self.show_tokenization = show_tokenization

        # Light mode
        self.camera.background_color = WHITE
        Text.set_default(color=BLACK)
        Tex.set_default(color=BLACK)

    def construct(self):

        title1 = Text("The Transformer as a function").move_to(UP * 2).scale(0.6)
        title2 = Text("Given word sequence, return next word").move_to(UP * 2).scale(0.6)
        title3 = Text("...with tokenization").move_to(UP * 2).scale(0.6)

        p1 = Tex("$Transformer($")
        p2 = Tex("$X$").set_color(BLUE_E)
        p3 = Tex("$) \\rightarrow$")
        p4 = Tex("$Y$").set_color(GREEN_E)
        eq_group = VGroup(p1, p2, p3, p4)
        eq_group.arrange()

        sentence = "the robots will bring prosperity".split(' ')
        prompt = sentence[:-1]
        answer = sentence[-1]
        p2_words = MathTex('``' + '\\;'.join(prompt) + '"').shift(LEFT).set_color(BLUE_E)
        p4_word = MathTex(f'``{answer}"').next_to(p2_words, RIGHT * 5).set_color(GREEN_E)

        sentence_token_ids = [VOCAB[word] for word in sentence]
        p2_ids = MathTex(str(sentence_token_ids)).set_color(BLUE_E)
        p4_id = MathTex(str(VOCAB[answer])).next_to(p2_ids, RIGHT * 5).set_color(GREEN_E)

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
        positions = [str(x) for x in list(range(4))]
        super().__init__(' '.join(positions), POS_EMB, xlabel=f"T = {len(positions)}", ylabel="C = 786", quote_tokens=False)


class WordEmbeddings(TwoColumnMapping):

    def __init__(self):
        sentence = "the robots will bring"
        sentence_tokens = sentence.split(' ')
        # input_str = ' '.join([f'{word}({VOCAB[word]})' for word in sentence_tokens])
        super().__init__(sentence, WORD_EMB, xlabel=f"T = {len(sentence_tokens)}", ylabel="C = 786")

class Tokenization(TwoColumnMapping):

    def __init__(self):
        sentence = "aardvark apple box cardboard"
        super().__init__(sentence, VOCAB, xlabel="All\\; words", ylabel="Unique\\; ID", show_dot_row=True)


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

        



