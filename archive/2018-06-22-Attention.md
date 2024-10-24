---
layout: post
title: Attention in Neural Networks
comments: true
---

<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>


This post presents the brief explanation of the following papers on how Attention in Neural Networks has evolved overtime.

- [Sequence to sequence models](#seq2seq)
- Neural Machine Translation by jointly learning to align and translate
- Pointer Networks
- Order Matters: Sequence to sequence for sets
- Attention is all you need

<a name='seq2seq'></a>
## Sequence to sequence models
At an abstract level, seq2seq models consists of an _Encoder_ and a _Decoder_. In general sequence to sequence models, we have an input sequence say \\(X = \left( x_1, x_2, ..., x_{T_x}\right)\\). Each input of the sequence is processed sequentially to obtain the representation of the encoder's hidden state \\(h_t\\) for \\(t = 1, 2, ..., T_x\\) where \\(h_t = f(x_t, h_{t-1})\\). Each annotation \\(h_t\\) can be thought of as focussing it's attention over the word \\(x_t\\). Finally, the entire input sequence is encoded in to the vector \\(C\\) where
$$
C = q(\{h_1, h_2, ..., h_{T_x}\})
$$

In general, \\(f\\) is represented as an _LSTM_ and 
$$
q(\{h_1, h_2, ..., h_{T_x}\}) = h_{T_x}
$$

Here we are using only the last encoding. Well, theoreticaly this is the encoding of the entire input sequence.
Once we have the encoded vector \\(C\\) (also referred to as the context vector), which in essence has squeezed the entire input sequence \\(X\\) to a fixed vector representation, we can decode \\(C\\) to obtain the output sequence \\(Y = \left({y_1, y_2, ..., y_{T_y}}\right)\\). The decoder predicts the word \\(y_{t}\\) conditioned on the context vector \\(C\\) and all the predicted vectors so far i.e, \\(\\{y_1, y_2, ..., y_{t-1}\\}\\). The joint probability over \\(Y\\) is obtained by the product of the marginal conditional distributions. Mathematically,

$$
p(Y) = \prod\limits_{t=1}^{T_y}p(y_t | \{y_1, y_2, ..., y_{t-1}\}, C) \tag{1}
$$

where the probability on the RHS of the above equation is represented by a non-linear multi-layered neural network.

$$
p(y_t | \{y_1, y_2, ..., y_{t-1}\}, C) = g(y_{t-1}, s_t, C) \tag{2}
$$

where \\(s_t\\) is the hidden state representation of the decoder network at time step \\(t\\) and \\(y_t\\) is the output of decoder network at time step \\(t\\) and \\(C\\) as usual is the context vector.

$$
s_t = \phi(y_{t-1}, s_{t-1}, C) \tag{3}
$$

The disadvatange with this particular model is we are forcing the encoder to encode the entire input sequence in to a fixed vector representation. This would probably suffice for the problems such as classification but this encoding is insufficient in the problems where one has to predict sequences such as translation systems where we might need access to each of the hidden annotations of the encoder. This entire model, is illustrated in the figure 1.

Solving the above major setback of encoding the entire input sequence in to a fixed vector representaation is addressed in the paper **Neural Machine Translation by jointly leraning to align and translate**

<a name='neural_translate'></a>
## Attention in translation

In this newly proposed architecture illustrated in the figure 2, the conditional probability of the equation equation (2) is rewritten as follows:

$$
p(y_t | \{y_1, y_2, ..., y_{t-1}\}, X) = g(y_{t-1}, s_t, c_t) \tag{4}
$$

where \\(s_t\\) is the hidden state representation of the decoder at time step \\(t\\) computed as 

$$
s_t = f(y_{t-1}, s_{t-1}, c_t) \tag{5}
$$

unlike in a standard seq2seq model where the decoder's hidden state is given by equation 3. We now go through how the context vector \\(c_t\\) is computed.

### Encoder

This is represented as a bi-directional RNN.

#### Forward RNN
Read the input sequence \\(X = \left( x_1, x_2, ..., x_{T}\right)\\) and then compute the forward hidden states \\(\overrightarrow{h} = \left(\overrightarrow{h_1}, \overrightarrow{h_2}, ..., \overrightarrow{h_T} \right)\\)

#### Backward RNN
Read the input sequence \\(X = \left( x_T, x_{T-1}, ..., x_1\right)\\) and then compute the backward hidden states \\(\overleftarrow{h} = \left(\overleftarrow{h_1}, \overleftarrow{h_2}, ..., \overleftarrow{h_T} \right)\\)

Combining the above two forward and backward hidden states \\(\overrightarrow{h_t}\\) and \\(\overleftarrow{h_t} \\) we get the overall hidden state at time step \\(t\\) as \\(h_t\\) is obtained by concatenating both of them,

$$
h_t = \Big[{\overrightarrow{h_t}}^{T}; \overleftarrow{h}^{T}\Big]^{T}
$$

In essense each annotation \\(h_t\\) now can summarize not the only the preceeding words but also the following words. Further each annotation \\(h_j\\) focusses its attention around the input sequence \\(x_j\\)

### Decoder

This is where the major difference comes in to play. Probability distribution over the predicted word \\(y_t\\) is conditioned on a different context vector \\(c_t\\) for each \\(t\\) unlike in equation 2. Each context vector \\(c_t\\) depends upon sequence of annotations \\(h = \left(h_1, h_2, ..., h_T\right)\\). More precisely, \\(c_t\\) is the expectation over all the annotations \\(h\\) with probabilities \\(\alpha_{tj}\\) for \\(j = 1, 2, ..., T\\) given by

$$
c_t = \sum_{j = 1}^{T} \alpha_{tj}h_j \tag{6}
$$

where \\(\alpha_{tj}\\) can be interpreted as how well the target word \\(y_t\\) is aligned to or translated from the input \\(x_j\\) and is obtained as follows

$$
\alpha_{tj} = \frac{\exp(e_{tj})}{\sum_{k=1}^{T}\exp(e_{tk})}
$$

Further, \\(\alpha_{tj}\\) reflects the importance of annotation \\(h_j\\) wrt the previous hidden state of decoder \\(s_{t-1}\\) in predicting the next hidden state \\(s_t\\) and generating the word \\(y_t\\). \\(e_{tj}\\) is obtained as

$$
e_{tj} = a(s_{t-1}, h_j)
$$

where \\(a\\) is represented as a Neural Network. For eg: (where \\(V\\) a vector, \\(W_1, W_2\\) square matrices are the learnable parameters of the network):

$$
e_{tj} = V^{T}\tanh(W_1s_{t-1} + W_2h_j)
$$

Energies \\(e_{tj}\\) can be tought of as how well the inputs around the position \\(j\\) and the output at \\(t^{th}\\) time step match or can be interpreted as the importance of the encoded vector \\(h_j\\) in predicting the word \\(y_t\\). Note that \\(\alpha_{tj}\\) from the equation 6 intuitively implement a mechanism of attention in the decoder. The decoder decides parts of the source sentence to pay attention to.

Problem? Well yes, in this particular model the output dictionary size is fixed and this cannot be applied to the models where the output size varies depending upon the input size. This brings us to the next section which tackles this problem by introducing the notion of pointers.

## Pointer Networks

## Order matters : Sequence to sequence for sets

## Attention is all you need

- What we have seen so far is that, attention mechanism has focussed to learn the dependencies between the input and the output. Transformer further extends this approach in learning the intra-input and intra-output dependencies as well.

## Neural Turing Machines and beyond

## Dilated Causal convolutions
- cuasal
