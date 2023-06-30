<script lang="ts">
  import Note from '$lib/components/note.svelte'
  import Video from '$lib/components/video.svelte'

  import TransformerFunc from '$lib/animations/media/videos/gen/1080p60/TransformerFunc.mp4'
  import TransformerFuncImg from '$lib/animations/media/images/gen/TransformerFuncImg.png'
  import Vocabulary from '$lib/animations/media/images/gen/Vocabulary_ManimCE_v0.17.3.png'

  import WordEmbeddings from '$lib/animations/media/videos/gen/1080p60/WordEmbeddings.mp4'
  import PositionEmbeddings from '$lib/animations/media/videos/gen/1080p60/PositionEmbeddings.mp4'
  import PreparingEmbeddings from '$lib/animations/media/videos/gen/1080p60/PreparingEmbeddings.mp4'

  import QKV from '$lib/animations/media/videos/gen/1080p60/QueryKeyValue.mp4'
  import SplittingHeads from '$lib/animations/media/videos/gen/1080p60/SplittingHeads.mp4'
  import SelfAttn from '$lib/animations/media/videos/gen/1080p60/SelfAttn.mp4'
  import SelfAttnPt2 from '$lib/animations/media/videos/gen/1080p60/SelfAttnPt2.mp4'
  import SelfAttnPt3 from '$lib/animations/media/videos/gen/1080p60/SelfAttnPt3.mp4'
  import FeedFwd from '$lib/animations/media/videos/gen/1080p60/FeedFwd.mp4'
  import GoingDeeper from '$lib/animations/media/videos/gen/1080p60/GoingDeeper.mp4'
  import Prediction from '$lib/animations/media/videos/gen/1080p60/Prediction.mp4'
  import GeneratingText from '$lib/animations/media/videos/gen/1080p60/GeneratingText.mp4'
</script>

<main class="flex justify-center items-start h-screen pt-20 2xl:pt-40">
  <div class="max-w-4xl px-4 py-4">
    <article
      class="prose-lg prose-h1:font-headline prose-headings:font-subheadline prose-h4:font-para prose-p:font-para prose-li:font-para"
    >
      <h1 class="text-center">The Animated Transformer</h1>
      <h6 class="text-center">
        by <a
          href="https://www.linkedin.com/in/sampathpraveen/"
          class="text-blue-500 font-bold underline"
          target="_blank">Praveen Sampath</a
        >
      </h6>

      <section>
        <p>
          The Transformer is foundational to the recent advancements in large language models
          (LLMs). In this article, we will attempt to unravel some of its inner workings and
          hopefully gain some insight into how these models function.
        </p>
        <p>
          The only prerequisite for following along with this article is a basic understanding of
          linear algebra - if you know how to multiply matrices, you are good to go.
        </p>
      </section>

      <h4 class="font-bold mb-20">Let's begin!</h4>

      <section>
        <h2>What is a Transformer?</h2>
        <p>
          The Transformer is a machine learning model for sequence modeling. Given a sequence of <i
            >things</i
          >, the model can predict what the next <i>thing</i>
          in the sequence might be. In this article, we will look at word sequence prediction, but you
          can apply Transformers to any kind of sequential data.
        </p>

        <div>
          <p>
            As an example, take the following phrase that we would like to complete (a.k.a. the
            &ldquo;prompt&rdquo;):
          </p>

          <p class="text-center">
            <code>&ldquo;The robots will bring _________&rdquo;</code>
          </p>
        </div>

        <p>
          Conceptually, we might think of the Transformer as a function that operates on this phrase
          as input:
        </p>

        <Video src={TransformerFunc} />
        <Note>Tap/click to pause animations, hold and drag to seek.</Note>

        <p>
          The input sentence is converted to a sequence of tokens (read: uniquely identifying
          numbers) before being passed as input to the model. The reason for this will become
          evident in the next section, where we will discuss how the input to the model is prepared.
        </p>

        <div class="flex flex-col justify-center 2xl:flex-row align-top">
          <p>
            The model parameters, denoted by &theta;, are the set of weights of the model that are
            tuned to the training data. We will take a peek into what &theta; contains, in each of
            the following sections.
          </p>
        </div>
      </section>

      <section>
        <h2>Tokenization</h2>

        <p>
          The Transformer operates on sequences, and so as a first step, we need to tokenize the
          given English phrase to a sequence of tokens that can be passed as input to the model. One
          obvious approach is to treat each word in the sentence as a token.
        </p>

        <p>
          The model doesn't understand words, it identifies tokens using a unique number assigned to
          each token. To achieve this, we assign a unique number to each word in our dictionary:
        </p>

        <img alt="vocabulary" src={Vocabulary} />

        <p>
          The call to the Transformer for our input sequence <code
            >&ldquo;the robots will bring&rdquo;</code
          > might then look something like:
        </p>
        <img alt="vocabulary" src={TransformerFuncImg} />

        <p>
          (where token number 2532 is the model's best guess for the token that might appear next in
          the sequence)
        </p>

        <p>
          We are now ready to pass our input to the Transformer. In the next several sections, we
          will walk through all the major steps involved in getting to the desired output: the next
          word in the sequence.
        </p>
      </section>

      <Note>
        In this article we only discuss the architecture of auto-regressive, decoder-only
        Transformers, like the GPT family of models from OpenAI. For simplicity, we do not consider
        encoder-decoder architectures such as the one described in the seminal &ldquo;Attention is
        All You Need&rdquo; 2017 paper.
        <p>
          We also do not cover model training, i.e., how the model parameters for the Transformer
          are tweaked and tuned to data (doing so would have made this already very long article
          unbearably long). Instead we take a trained model, and walk through the computation that
          takes place during inference.
        </p>
        Most of the details described here come from the excellent
        <a
          href="https://github.com/karpathy/nanoGPT/blob/master/model.py"
          target="_blank"
          class="underline">nanoGPT implementation by Andrej Karpathy</a
        >, which is roughly equivalent to GPT-2.
      </Note>
      <section>
        <h2>1. Embeddings: Numbers Speak Louder than Words</h2>

        <p>
          For each token, the Transformer maintains a vector called an &ldquo;embedding&rdquo;. An
          embedding aims to capture the semantic meaning of the token - similar tokens have similar
          embeddings.
        </p>

        <p>The input tokens are mapped to their corresponding embeddings:</p>
        <Video src={WordEmbeddings} />

        <p>
          Our transformer has embedding vectors of length 768. All the embeddings can be packed
          together in a single
          <code>T &times; C</code> matrix, where
          <code>T = 4</code> is the number of input tokens, and
          <code>C = 768</code>, the size of each embedding.
        </p>

        <p>
          In order to capture the significance of the position of a token within a sequence, the
          Transformer also maintains embeddings for each position. Here, we fetch embeddings for
          positions 0 to 3, since we only have 4 tokens:
        </p>
        <Video src={PositionEmbeddings} />

        <p>
          Finally, these two <code>T &times; C</code> matrices are added together to obtain a position-dependent
          embedding for each token:
        </p>
        <Video src={PreparingEmbeddings} />

        <p>
          The token and position embeddings are all part of &theta;, the model parameters, which
          means they are tuned during model training.
        </p>
      </section>

      <section>
        <h2>2. Queries, keys and values</h2>

        <p>
          The Transformer then computes three vectors for each of the <code>T</code> vectors (each
          of row in the <code>T &times; C</code> matrix from the previous section): &ldquo;query&rdquo;,
          &ldquo;key&rdquo; and &ldquo;value&rdquo; vectors. This is done by way of three linear transformations
          (i.e., multiplying with a weight matrix):
        </p>
        <Video src={QKV} />

        <p>
          The weight matrices that produce <code>Q, K, V</code> matrices are all part of &theta;.
        </p>

        <p>
          The query, key and value vectors for each token are packed together into <code
            >T &times; C</code
          > matrices, just like the input embedding matrix. These vectors are the primary participants
          involved in the main event which is coming up shortly: self-attention.
        </p>

        <Note>
          <p>
            But what are these queries, keys and values? One way to think about these vectors is
            using the following analogy:
          </p>

          <p>
            Imagine you have a database of images with their text descriptions, and you would like
            to build an image search engine. Users will search by providing some text
            (the <b>query</b>), which will be matched against the text descriptions in your database
            (the <b>key</b>), and the final result is the image itself (the <b>value</b>). 
            Only those images (values) whose corresponding text description (key) best matches 
            the user input (query) will be returned in the search results.
          </p>

          <p>
            Self-attention works in a similar manner - the tokens in the user input are trying to &ldquo;query&rdquo;
            other tokens to find the ones they should be paying attention to.
          </p>
        </Note>
      </section>

      <section>
        <h2>3. Two heads are better than one</h2>

        <p>
          To recap: so far, starting from our input <code>T &times; C</code> matrix containing the
          token + position embeddings, we have computed three <code>T &times; C</code> matrices: (1)
          the Query matrix <code>Q</code>, (2) the Key matrix <code>K</code>, and (3) the Value
          matrix <code>V</code>.
        </p>

        <p>
          The Transformer then splits these matrics into multiple so-called &ldquo;heads&rdquo;:
        </p>
        <Video src={SplittingHeads} />

        <p>
          Here, we see the <code>Q</code> matrix being split length-wise into twelve
          <code>(T &times; H)</code>
          heads. Since <code>Q</code> has 768 columns, each head has 64 columns.
        </p>

        <p>
          Self-attention operates independently within each head, and it does so in parallel. In
          other words, the first head of the Query matrix only interacts with the first heads of the
          Key and Value matrices. There is no interaction <i>between</i> different heads.
        </p>

        <p>
          The idea behind splitting into multiple heads is to afford greater freedom to the
          Transformer to capture different characteristics of the input embeddings. E.g., the first
          head might specialize in capturing the part-of-speech relationships, and another might
          focus on semantic meaning, etc.
        </p>
      </section>

      <section>
        <h2>4. Time to pay attention</h2>

        <p>
          Self-attention, as we've alluded to earlier, is the core idea behind the Transformer
          model.
        </p>

        <p>
          We first compute an &ldquo;attention scores&rdquo; matrix by multiplying the query and key
          matrices (note that we are only looking at the first head here, but the same operation
          occurs for all heads):
        </p>
        <Video src={SelfAttn} />

        <p>
          This matrix tells us how much attention, or weightage, a particular token needs to pay to
          every other token in the sequence for producing its output, i.e., prediction for the next
          token. E.g., the token "bring" has an attention score of 0.3 for the token "robot" (row 4,
          column 2 in matrix <code>A<sub>1</sub></code>).
        </p>
      </section>

      <section>
        <h2>5. Applying attention</h2>

        <p>
          The attention score for a token needs to be <i>masked</i> if it occurs earlier in the
          sequence for a given target token. E.g., in our input phrase:
          <code>&ldquo;the robots will bring _____&rdquo;</code> it makes sense for the token
          <code>&ldquo;bring&rdquo;</code>
          to pay attention to the token <code>&ldquo;robots&rdquo;</code>, but not vice-versa,
          because a token should not be allowed to look to the future tokens for making a prediction
          of its next token.
        </p>

        <p>
          So we hide the upper-right triangle of the square matrix <code>A<sub>1</sub></code>,
          effectively setting the attention score to 0.
        </p>

        <p>
          We then bring the third actor onto the stage, the Value matrix <code>V</code>:
        </p>
        <Video src={SelfAttnPt2} />

        <p>
          The output for the token &ldquo;robots&rdquo; is a weighted sum of the Value vectors for
          the previous token &ldquo;the&rdquo; and itself. Specifically, in this case, it applies a
          47% weight to the former, and 53% weight to its own Value vector (work out the matrix
          multiplication between <code>A<sub>1</sub>(T &times T)</code> and
          <code>V<sub>1</sub>(T &times H)</code> to convince yourself that this is true). The outputs
          for all other tokens is computed similarly.
        </p>
      </section>

      <section>
        <h2>5. Putting all heads together</h2>

        <p>
          The final output for each head of self-attention is a matrix <code>Y</code> of dimensions
          <code>T &times H (T = 4, H = 64)</code>.
        </p>

        <p>
          Having computed the output embeddings for all tokens across all 12 heads, we now combine
          the individual <code>T &times; H</code> into a single matrix of dimension
          <code>T &times; C</code>, by simply stacking them side-by-side:
        </p>
        <Video src={SelfAttnPt3} />

        <p>
          64 embedding dims per head <code>(H)</code>&times; 12 heads = 768, the original size of
          our input embeddings <code>(C)</code>.
        </p>

        <Note>
          Sections 2 - 5 collectively describe what happens in a single <b>self-attention unit</b>.
          <br /><br />
          The input and output of a single self-attention unit are both <code>T &times; C</code> dimension
          matrices.
        </Note>
      </section>

      <section>
        <h2>6. Feed forward</h2>

        <p>
          Everything we have done up to this point has involved only linear operations - i.e.,
          matrix multiplications. This is not sufficient to capture complex relationships between
          tokens, so the Transformer introduces a single hidden-layer neural network (also referred
          to as a <b>feed forward network</b>, or a <b>multi-layer perceptron (MLP)</b>), with
          non-linearity.
        </p>
        <Video src={FeedFwd} />

        <p>
          The <code>C</code> length row vectors are transformed to vectors of length
          <code>(4 * C)</code>
          by way of a linear transform, a non-linear function like ReLU is applied, and finally the vectors
          are linearly transformed back to vectors of length <code>C</code>.
        </p>

        <p>All the weight matrices involved in the feed forward network are part of &theta;.</p>
      </section>

      <section>
        <h2>7. We need to go deeper</h2>

        <p>
          All the steps in Sections 2 to 6 above constitute a single <b>Transformer block</b>. Each
          block takes as input a <code>T &times; C</code> matrix, and outputs a
          <code>T &times; C</code> matrix.
        </p>

        <p>
          In order to arm our Transformer model with the ability to capture complex relationships
          between words, many such blocks are stacked together in sequence:
        </p>
        <Video src={GoingDeeper} />
      </section>

      <section>
        <h2>8. Making a prediction</h2>

        <p>Finally, we are ready to make a prediction:</p>
        <Video src={Prediction} />

        <p>
          The last output of the last block in the Transformer will give us a <code>C</code> length
          vector for each of our input tokens. Since we only care about what comes after the last
          token, &ldquo;bring&rdquo;, we look at its vector. A linear transform on this vector -
          multiplying with another weight matrix of dimensions <code>V &times C</code>, where
          <code>V</code>
          is the total number of words in our dictionary - will give us a vector of length
          <code>V</code>.
        </p>

        <p>
          This vector when normalized gives us a probability distribution over every word in our
          dictionary, which allows us to the pick the one with the highest probability as the next
          token.
        </p>

        <p>
          In this case, our Transformer has assigned a probability of 92% on
          &ldquo;prosperity&rdquo; being the next token, while there's only a 10% chance of
          &ldquo;destruction&rdquo;, so our completed sentence now reads: &ldquo;the robots will
          bring prosperity&rdquo;. I suppose we can now rest easy with the knowledge that AI's
          imminent takeover of human civilization promises a future of prosperity and well-being,
          rather than death and destruction.
        </p>
      </section>

      <section>
        <h2>9. Text Generator Go Brrr</h2>

        <p>Now that we can predict the next token, we can generate text, one token at a time:</p>
        <Video src={GeneratingText} />

        <p>
          The first token that the model produces is added to the prompt and fed back into it to
          produce the second token, which is then fed back into it to produce the third, and so on. The
          Transformer has a limit on the maximum number of tokens (N) that it can take as input, and
          therefore as the number of generated tokens increases, eventually we need to either cap the
          number of tokens to keep only the last N, or devise some other technique for shortening
          the prompt without losing information from the oldest tokens.
        </p>
      </section>

      <section>
        <h2>And that's it!</h2>

        <p>
          If you made it all the way to the end, well done! I hope this was worthwhile and
          contributed in some way to your understanding of LLMs.
        </p>

        <Note>
          To focus on the most important aspects of a Transformer's implementation, several details
          were intentionally left out. Some of these include: layer normalization, dropout, residual
          connections, etc.
          <br /><br />
          I highly recommend reading the full
          <a
            href="https://github.com/karpathy/nanoGPT/blob/master/model.py"
            target="_blank"
            class="underline">nanoGPT implementation</a
          > to get all the details.
        </Note>

        <p>
          This project was made with <a
            href="https://github.com/ManimCommunity/manim"
            target="_blank"
            class="underline text-blue-500 font-bold">Manim</a
          >, a Python library for mathematical animations, created by Grant Sanderson who runs the
          YouTube channel
          <a
            target="_blank"
            class="underline text-blue-500 font-bold"
            href="https://www.youtube.com/c/3blue1brown">3Blue1Brown</a
          > (which you should definitely check out, BTW).
        </p>
      </section>

      <div class="h-10 mb-40" />
    </article>
  </div>
</main>
