<script lang="ts">
  import Note from '../lib/components/note.svelte';
</script>

<main class="flex justify-center items-start h-screen pt-40">
  <div class="max-w-xl px-4 py-4">
    <div class="flex flex-col 2xl:flex-row">

      <div class="w-full 2xl:w-1/2">
        <!-- Right Column -->
        <article class="prose-lg prose-h1:font-headline prose-headings:font-subheadline prose-p:font-para">
          <h1 class="text-center">The Animated Transformer</h1>
          <p>
            The Transformer is foundational to the recent advancements in large language models (LLMs).
            In this article, I attempt to unravel some of its inner workings and leave you with a deeper,
            more intuitive understanding of how and why these models work.
          </p>
          <p>
            The ideal target audience for this article is someone who has a basic 
            high-school-level understanding of linear algebra (e.g., operating on matrices and vectors) -
            although if you don't, fret not! You should still be able to follow along. Some understanding of
            basic machine learning concepts will help, but is not strictly required.
          </p>

          <p>
            There have been many excellent articles written on the Transformer, but I could fully understand
            all the important details only after I read through the following implementation of a small-scale 
            GPT model for character-level text prediction: nanoGPT.
          </p>

          <h4 class="font-bold mb-20">Let's begin!</h4>

          <h2>What is a Transformer anyway?</h2>
          <p>
            The Transformer is a machine learning model that can do sequence prediction:
            given a sequence of <i>things</i>, the model predicts what the next <i>thing</i>
            in the sequence might be. The <i>thing sequence</i> could be whatever you fancy - sequences of musical notes 
            to generate music, sequences of DNA base-pairs for predicting gene expression; or, as in the
            case of language models, sequences of words. 
          </p>

          <Note>
            TODO: We are talking about generative decoder only transformers here. Mention how
            this relates to models like GPT-4, ChatGPT, etc.
          </Note>

          <p>
            Focusing specifically on text sequences, we might picture the transformer as a 
            <i>function</i> that takes as input a given sequence of words, and outputs a word that
            is most likely to occur next:
          </p>

          <code class="border-solid border-red-500 border-2">
            fig 1
          </code>

          <p>
            But how do we make a function that predicts the next word? The answer
            - as you might already know - is "model training". But what is "training" and how does it work?
          </p>
          <p>
            Before we talk about model training, I think it makes sense to first peer into this
            next-word-predicting Transformer function and understand what it looks like.
          </p>

          <h2>Tokenization: Preparing the Input</h2>

          <p>
            As we talked about in the previous section, our Transformer expects as input a sequence of words.
            Continuing from our earlier example, say we want to find out what comes next in the following
            piece of text:
          </p>

          <code class="border-solid border-red-500 border-2">
            fig 2
          </code>

          <p>
            The first step is to break this text into a sequence of words, and assign a unique number
            to each word (why we do this will become clearer in the next section). One way to do this is to assign
            numbers based on alphabetic order - so "aardvark" might be assigned 1, while "cardboard" might be assigned 134:
          </p>

          <code class="border-solid border-red-500 border-2">
            fig 3
          </code>

          <p>
            We are now ready to pass our input to the Transformer. In the next several sections, we will walk
            through all the major steps involved in getting to the desired output: the next word in the sequence. 
          </p>

          <h2>1. Embeddings: Numbers Speak Louder than Words</h2>

          <p>The </p>

          <h2>Pay attention!</h2>

        </article>
      </div>

      <div class="w-full 2xl:w-1/2 mb-4 md:mb-0">
        <!-- Left Column -->
      </div>


    </div>
  </div>
</main>
