from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# We disable pylint because we need python3 compatibility.
from six.moves import xrange    # pylint: disable=redefined-builtin
from six.moves import zip         # pylint: disable=redefined-builtin

from tensorflow.python import shape
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest
import tensorflow as tf
import rnn_cell as my_rnn_cell
# TODO(ebrevdo): Remove once _linear is fully deprecated.
linear = rnn_cell._linear    # pylint: disable=protected-access


def _extract_argmax_and_embed(embedding, num_symbols, output_projection=None,
                                                            update_embedding=True):
    """Get a loop_function that extracts the previous symbol and embeds it.

    Args:
        embedding: embedding tensor for symbols.
        output_projection: None or a pair (W, B). If provided, each fed previous
            output will first be multiplied by W and added B.
        update_embedding: Boolean; if False, the gradients will not propagate
            through the embeddings.

    Returns:
        A loop function.
    """
    def loop_function(prev, _):
        #if output_projection is not None:
        #    prev = nn_ops.xw_plus_b(
        #            prev, output_projection[0], output_projection[1])
        #prev_symbol = math_ops.argmax(prev, 1)
        prev_symbol = math_ops.argmax(array_ops.split_v(prev, [2, num_symbols-2], 1)[1], 1) + 2
        # Note that gradients will not propagate through the second parameter of
        # embedding_lookup.
        emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
        if not update_embedding:
            emb_prev = array_ops.stop_gradient(emb_prev)
        return emb_prev
    return loop_function

def _extract_beam_search(embedding, beam_size, num_symbols, embedding_size, output_projection=None, update_embedding=True):
    """Get a loop_function that extracts the previous symbol and embeds it.
    Args:
        embedding: embedding tensor for symbols.
        output_projection: None or a pair (W, B). If provided, each fed previous
            output will first be multiplied by W and added B.
        update_embedding: Boolean; if False, the gradients will not propagate
            through the embeddings.
    Returns:
        A loop function.
    """
    def loop_function(prev, i, log_beam_probs, beam_path, beam_symbols, beam_results):
        #if output_projection is not None:
        #    prev = nn_ops.xw_plus_b(
        #            prev, output_projection[0], output_projection[1])
        # prev= prev.get_shape().with_rank(2)[1]
        prev = array_ops.split_v(prev, [2, num_symbols-2], 1)[1]
        probs = tf.log(prev+1e-12)

        if i > 1:

            probs = tf.reshape(probs + log_beam_probs[-1],
                                                             [-1, beam_size * (num_symbols - 2)])

        best_probs, indices = tf.nn.top_k(probs, beam_size * 2)
        indices = tf.stop_gradient(tf.squeeze(tf.reshape(indices, [-1, 1])))
        best_probs = tf.stop_gradient(tf.reshape(best_probs, [-1, 1]))

        symbols = indices % (num_symbols - 2) + 2 # Which word in vocabulary.

        beam_parent = indices // (num_symbols - 2) # Which hypothesis it came from.

        partition = tf.cast(tf.cast(symbols-2, tf.bool), tf.int32)

        prob_group = tf.dynamic_partition(best_probs, partition, 2)
        symbols_group = tf.dynamic_partition(symbols, partition, 2)
        parent_group = tf.dynamic_partition(beam_parent, partition, 2)
        
        beam_results.append([prob_group[0], symbols_group[0], parent_group[0]])

        _probs = prob_group[1][:beam_size]
        _symbols = symbols_group[1][:beam_size]
        _parents = parent_group[1][:beam_size]

        beam_symbols.append(_symbols)
        beam_path.append(_parents)
        log_beam_probs.append(_probs)

        # Note that gradients will not propagate through the second parameter of
        # embedding_lookup.

        emb_prev = embedding_ops.embedding_lookup(embedding, _symbols)
        emb_prev    = tf.reshape(emb_prev,[beam_size,embedding_size])
        # emb_prev = embedding_ops.embedding_lookup(embedding, symbols)
        if not update_embedding:
            emb_prev = array_ops.stop_gradient(emb_prev)
        return emb_prev
    return loop_function



def rnn_decoder(decoder_inputs, initial_state, cell, loop_function=None,
                                scope=None):
    """RNN decoder for the sequence-to-sequence model.

    Args:
        decoder_inputs: A list of 2D Tensors [batch_size x input_size].
        initial_state: 2D Tensor with shape [batch_size x cell.state_size].
        cell: rnn_cell.RNNCell defining the cell function and size.
        loop_function: If not None, this function will be applied to the i-th output
            in order to generate the i+1-st input, and decoder_inputs will be ignored,
            except for the first element ("GO" symbol). This can be used for decoding,
            but also for training to emulate http://arxiv.org/abs/1506.03099.
            Signature -- loop_function(prev, i) = next
                * prev is a 2D Tensor of shape [batch_size x output_size],
                * i is an integer, the step number (when advanced control is needed),
                * next is a 2D Tensor of shape [batch_size x input_size].
        scope: VariableScope for the created subgraph; defaults to "rnn_decoder".

    Returns:
        A tuple of the form (outputs, state), where:
            outputs: A list of the same length as decoder_inputs of 2D Tensors with
                shape [batch_size x output_size] containing generated outputs.
            state: The state of each cell at the final time-step.
                It is a 2D Tensor of shape [batch_size x cell.state_size].
                (Note that in some cases, like basic RNN cell or GRU cell, outputs and
                 states can be the same. They are different for LSTM cells though.)
    """
    with variable_scope.variable_scope(scope or "rnn_decoder"):
        state = initial_state
        outputs = []
        prev = None
        for i, inp in enumerate(decoder_inputs):
            if loop_function is not None and prev is not None:
                with variable_scope.variable_scope("loop_function", reuse=True):
                    inp = loop_function(prev, i)
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()
            output, state = cell(inp, state)
            outputs.append(output)
            if loop_function is not None:
                prev = output
    return outputs, state


def basic_rnn_seq2seq(
        encoder_inputs, decoder_inputs, cell, dtype=dtypes.float32, scope=None):
    """Basic RNN sequence-to-sequence model.

    This model first runs an RNN to encode encoder_inputs into a state vector,
    then runs decoder, initialized with the last encoder state, on decoder_inputs.
    Encoder and decoder use the same RNN cell type, but don't share parameters.

    Args:
        encoder_inputs: A list of 2D Tensors [batch_size x input_size].
        decoder_inputs: A list of 2D Tensors [batch_size x input_size].
        cell: rnn_cell.RNNCell defining the cell function and size.
        dtype: The dtype of the initial state of the RNN cell (default: tf.float32).
        scope: VariableScope for the created subgraph; default: "basic_rnn_seq2seq".

    Returns:
        A tuple of the form (outputs, state), where:
            outputs: A list of the same length as decoder_inputs of 2D Tensors with
                shape [batch_size x output_size] containing the generated outputs.
            state: The state of each decoder cell in the final time-step.
                It is a 2D Tensor of shape [batch_size x cell.state_size].
    """
    with variable_scope.variable_scope(scope or "basic_rnn_seq2seq"):
        _, enc_state = rnn.rnn(cell, encoder_inputs, dtype=dtype)
        return rnn_decoder(decoder_inputs, enc_state, cell)


def tied_rnn_seq2seq(encoder_inputs, decoder_inputs, cell,
                                         loop_function=None, dtype=dtypes.float32, scope=None):
    """RNN sequence-to-sequence model with tied encoder and decoder parameters.

    This model first runs an RNN to encode encoder_inputs into a state vector, and
    then runs decoder, initialized with the last encoder state, on decoder_inputs.
    Encoder and decoder use the same RNN cell and share parameters.

    Args:
        encoder_inputs: A list of 2D Tensors [batch_size x input_size].
        decoder_inputs: A list of 2D Tensors [batch_size x input_size].
        cell: rnn_cell.RNNCell defining the cell function and size.
        loop_function: If not None, this function will be applied to i-th output
            in order to generate i+1-th input, and decoder_inputs will be ignored,
            except for the first element ("GO" symbol), see rnn_decoder for details.
        dtype: The dtype of the initial state of the rnn cell (default: tf.float32).
        scope: VariableScope for the created subgraph; default: "tied_rnn_seq2seq".

    Returns:
        A tuple of the form (outputs, state), where:
            outputs: A list of the same length as decoder_inputs of 2D Tensors with
                shape [batch_size x output_size] containing the generated outputs.
            state: The state of each decoder cell in each time-step. This is a list
                with length len(decoder_inputs) -- one item for each time-step.
                It is a 2D Tensor of shape [batch_size x cell.state_size].
    """
    with variable_scope.variable_scope("combined_tied_rnn_seq2seq"):
        scope = scope or "tied_rnn_seq2seq"
        _, enc_state = rnn.rnn(
                cell, encoder_inputs, dtype=dtype, scope=scope)
        variable_scope.get_variable_scope().reuse_variables()
        return rnn_decoder(decoder_inputs, enc_state, cell,
                                             loop_function=loop_function, scope=scope)


def embedding_rnn_decoder(decoder_inputs,
                                                    initial_state,
                                                    cell,
                                                    num_symbols,
                                                    embedding_size,
                                                    output_projection=None,
                                                    feed_previous=False,
                                                    update_embedding_for_previous=True,
                                                    scope=None):
    """RNN decoder with embedding and a pure-decoding option.

    Args:
        decoder_inputs: A list of 1D batch-sized int32 Tensors (decoder inputs).
        initial_state: 2D Tensor [batch_size x cell.state_size].
        cell: rnn_cell.RNNCell defining the cell function.
        num_symbols: Integer, how many symbols come into the embedding.
        embedding_size: Integer, the length of the embedding vector for each symbol.
        output_projection: None or a pair (W, B) of output projection weights and
            biases; W has shape [output_size x num_symbols] and B has
            shape [num_symbols]; if provided and feed_previous=True, each fed
            previous output will first be multiplied by W and added B.
        feed_previous: Boolean; if True, only the first of decoder_inputs will be
            used (the "GO" symbol), and all other decoder inputs will be generated by:
                next = embedding_lookup(embedding, argmax(previous_output)),
            In effect, this implements a greedy decoder. It can also be used
            during training to emulate http://arxiv.org/abs/1506.03099.
            If False, decoder_inputs are used as given (the standard decoder case).
        update_embedding_for_previous: Boolean; if False and feed_previous=True,
            only the embedding for the first symbol of decoder_inputs (the "GO"
            symbol) will be updated by back propagation. Embeddings for the symbols
            generated from the decoder itself remain unchanged. This parameter has
            no effect if feed_previous=False.
        scope: VariableScope for the created subgraph; defaults to
            "embedding_rnn_decoder".

    Returns:
        A tuple of the form (outputs, state), where:
            outputs: A list of the same length as decoder_inputs of 2D Tensors. The
                output is of shape [batch_size x cell.output_size] when
                output_projection is not None (and represents the dense representation
                of predicted tokens). It is of shape [batch_size x num_decoder_symbols]
                when output_projection is None.
            state: The state of each decoder cell in each time-step. This is a list
                with length len(decoder_inputs) -- one item for each time-step.
                It is a 2D Tensor of shape [batch_size x cell.state_size].

    Raises:
        ValueError: When output_projection has the wrong shape.
    """
    with variable_scope.variable_scope(scope or "embedding_rnn_decoder") as scope:
        if output_projection is not None:
            dtype = scope.dtype
            proj_weights = ops.convert_to_tensor(output_projection[0], dtype=dtype)
            proj_weights.get_shape().assert_is_compatible_with([None, num_symbols])
            proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
            proj_biases.get_shape().assert_is_compatible_with([num_symbols])

        embedding = variable_scope.get_variable("embedding",
                                                                                        [num_symbols, embedding_size])
        loop_function = _extract_argmax_and_embed(
                embedding, output_projection,
                update_embedding_for_previous) if feed_previous else None
        emb_inp = (
                embedding_ops.embedding_lookup(embedding, i) for i in decoder_inputs)
        return rnn_decoder(emb_inp, initial_state, cell,
                                             loop_function=loop_function)


def embedding_rnn_seq2seq(encoder_inputs,
                                                    decoder_inputs,
                                                    cell,
                                                    num_encoder_symbols,
                                                    num_decoder_symbols,
                                                    embedding_size,
                                                    output_projection=None,
                                                    feed_previous=False,
                                                    dtype=None,
                                                    scope=None):
    """Embedding RNN sequence-to-sequence model.

    This model first embeds encoder_inputs by a newly created embedding (of shape
    [num_encoder_symbols x input_size]). Then it runs an RNN to encode
    embedded encoder_inputs into a state vector. Next, it embeds decoder_inputs
    by another newly created embedding (of shape [num_decoder_symbols x
    input_size]). Then it runs RNN decoder, initialized with the last
    encoder state, on embedded decoder_inputs.

    Args:
        encoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
        decoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
        cell: rnn_cell.RNNCell defining the cell function and size.
        num_encoder_symbols: Integer; number of symbols on the encoder side.
        num_decoder_symbols: Integer; number of symbols on the decoder side.
        embedding_size: Integer, the length of the embedding vector for each symbol.
        output_projection: None or a pair (W, B) of output projection weights and
            biases; W has shape [output_size x num_decoder_symbols] and B has
            shape [num_decoder_symbols]; if provided and feed_previous=True, each
            fed previous output will first be multiplied by W and added B.
        feed_previous: Boolean or scalar Boolean Tensor; if True, only the first
            of decoder_inputs will be used (the "GO" symbol), and all other decoder
            inputs will be taken from previous outputs (as in embedding_rnn_decoder).
            If False, decoder_inputs are used as given (the standard decoder case).
        dtype: The dtype of the initial state for both the encoder and encoder
            rnn cells (default: tf.float32).
        scope: VariableScope for the created subgraph; defaults to
            "embedding_rnn_seq2seq"

    Returns:
        A tuple of the form (outputs, state), where:
            outputs: A list of the same length as decoder_inputs of 2D Tensors. The
                output is of shape [batch_size x cell.output_size] when
                output_projection is not None (and represents the dense representation
                of predicted tokens). It is of shape [batch_size x num_decoder_symbols]
                when output_projection is None.
            state: The state of each decoder cell in each time-step. This is a list
                with length len(decoder_inputs) -- one item for each time-step.
                It is a 2D Tensor of shape [batch_size x cell.state_size].
    """
    with variable_scope.variable_scope(scope or "embedding_rnn_seq2seq") as scope:
        if dtype is not None:
            scope.set_dtype(dtype)
        else:
            dtype = scope.dtype

        # Encoder.
        encoder_cell = rnn_cell.EmbeddingWrapper(
                cell, embedding_classes=num_encoder_symbols,
                embedding_size=embedding_size)
        _, encoder_state = rnn.rnn(encoder_cell, encoder_inputs, dtype=dtype)

        # Decoder.
        if output_projection is None:
            cell = rnn_cell.OutputProjectionWrapper(cell, num_decoder_symbols)

        if isinstance(feed_previous, bool):
            return embedding_rnn_decoder(
                    decoder_inputs,
                    encoder_state,
                    cell,
                    num_decoder_symbols,
                    embedding_size,
                    output_projection=output_projection,
                    feed_previous=feed_previous)

        # If feed_previous is a Tensor, we construct 2 graphs and use cond.
        def decoder(feed_previous_bool):
            reuse = None if feed_previous_bool else True
            with variable_scope.variable_scope(
                    variable_scope.get_variable_scope(), reuse=reuse) as scope:
                outputs, state = embedding_rnn_decoder(
                        decoder_inputs, encoder_state, cell, num_decoder_symbols,
                        embedding_size, output_projection=output_projection,
                        feed_previous=feed_previous_bool,
                        update_embedding_for_previous=False)
                state_list = [state]
                if nest.is_sequence(state):
                    state_list = nest.flatten(state)
                return outputs + state_list

        outputs_and_state = control_flow_ops.cond(feed_previous,
                                                                                            lambda: decoder(True),
                                                                                            lambda: decoder(False))
        outputs_len = len(decoder_inputs)    # Outputs length same as decoder inputs.
        state_list = outputs_and_state[outputs_len:]
        state = state_list[0]
        if nest.is_sequence(encoder_state):
            state = nest.pack_sequence_as(structure=encoder_state,
                                                                        flat_sequence=state_list)
        return outputs_and_state[:outputs_len], state


def embedding_tied_rnn_seq2seq(encoder_inputs,
                                                             decoder_inputs,
                                                             cell,
                                                             num_symbols,
                                                             embedding_size,
                                                             num_decoder_symbols=None,
                                                             output_projection=None,
                                                             feed_previous=False,
                                                             dtype=None,
                                                             scope=None):
    """Embedding RNN sequence-to-sequence model with tied (shared) parameters.

    This model first embeds encoder_inputs by a newly created embedding (of shape
    [num_symbols x input_size]). Then it runs an RNN to encode embedded
    encoder_inputs into a state vector. Next, it embeds decoder_inputs using
    the same embedding. Then it runs RNN decoder, initialized with the last
    encoder state, on embedded decoder_inputs. The decoder output is over symbols
    from 0 to num_decoder_symbols - 1 if num_decoder_symbols is none; otherwise it
    is over 0 to num_symbols - 1.

    Args:
        encoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
        decoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
        cell: rnn_cell.RNNCell defining the cell function and size.
        num_symbols: Integer; number of symbols for both encoder and decoder.
        embedding_size: Integer, the length of the embedding vector for each symbol.
        num_decoder_symbols: Integer; number of output symbols for decoder. If
            provided, the decoder output is over symbols 0 to num_decoder_symbols - 1.
            Otherwise, decoder output is over symbols 0 to num_symbols - 1. Note that
            this assumes that the vocabulary is set up such that the first
            num_decoder_symbols of num_symbols are part of decoding.
        output_projection: None or a pair (W, B) of output projection weights and
            biases; W has shape [output_size x num_symbols] and B has
            shape [num_symbols]; if provided and feed_previous=True, each
            fed previous output will first be multiplied by W and added B.
        feed_previous: Boolean or scalar Boolean Tensor; if True, only the first
            of decoder_inputs will be used (the "GO" symbol), and all other decoder
            inputs will be taken from previous outputs (as in embedding_rnn_decoder).
            If False, decoder_inputs are used as given (the standard decoder case).
        dtype: The dtype to use for the initial RNN states (default: tf.float32).
        scope: VariableScope for the created subgraph; defaults to
            "embedding_tied_rnn_seq2seq".

    Returns:
        A tuple of the form (outputs, state), where:
            outputs: A list of the same length as decoder_inputs of 2D Tensors with
                shape [batch_size x output_symbols] containing the generated
                outputs where output_symbols = num_decoder_symbols if
                num_decoder_symbols is not None otherwise output_symbols = num_symbols.
            state: The state of each decoder cell at the final time-step.
                It is a 2D Tensor of shape [batch_size x cell.state_size].

    Raises:
        ValueError: When output_projection has the wrong shape.
    """
    with variable_scope.variable_scope(
            scope or "embedding_tied_rnn_seq2seq", dtype=dtype) as scope:
        dtype = scope.dtype

        if output_projection is not None:
            proj_weights = ops.convert_to_tensor(output_projection[0], dtype=dtype)
            proj_weights.get_shape().assert_is_compatible_with([None, num_symbols])
            proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
            proj_biases.get_shape().assert_is_compatible_with([num_symbols])

        embedding = variable_scope.get_variable(
                "embedding", [num_symbols, embedding_size], dtype=dtype)

        emb_encoder_inputs = [embedding_ops.embedding_lookup(embedding, x)
                                                    for x in encoder_inputs]
        emb_decoder_inputs = [embedding_ops.embedding_lookup(embedding, x)
                                                    for x in decoder_inputs]

        output_symbols = num_symbols
        if num_decoder_symbols is not None:
            output_symbols = num_decoder_symbols
        if output_projection is None:
            cell = rnn_cell.OutputProjectionWrapper(cell, output_symbols)

        if isinstance(feed_previous, bool):
            loop_function = _extract_argmax_and_embed(
                    embedding, output_projection, True) if feed_previous else None
            return tied_rnn_seq2seq(emb_encoder_inputs, emb_decoder_inputs, cell,
                                                            loop_function=loop_function, dtype=dtype)

        # If feed_previous is a Tensor, we construct 2 graphs and use cond.
        def decoder(feed_previous_bool):
            loop_function = _extract_argmax_and_embed(
                embedding, output_projection, False) if feed_previous_bool else None
            reuse = None if feed_previous_bool else True
            with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                                                                 reuse=reuse):
                outputs, state = tied_rnn_seq2seq(
                        emb_encoder_inputs, emb_decoder_inputs, cell,
                        loop_function=loop_function, dtype=dtype)
                state_list = [state]
                if nest.is_sequence(state):
                    state_list = nest.flatten(state)
                return outputs + state_list

        outputs_and_state = control_flow_ops.cond(feed_previous,
                                                                                            lambda: decoder(True),
                                                                                            lambda: decoder(False))
        outputs_len = len(decoder_inputs)    # Outputs length same as decoder inputs.
        state_list = outputs_and_state[outputs_len:]
        state = state_list[0]
        # Calculate zero-state to know it's structure.
        static_batch_size = encoder_inputs[0].get_shape()[0]
        for inp in encoder_inputs[1:]:
            static_batch_size.merge_with(inp.get_shape()[0])
        batch_size = static_batch_size.value
        if batch_size is None:
            batch_size = array_ops.shape(encoder_inputs[0])[0]
        zero_state = cell.zero_state(batch_size, dtype)
        if nest.is_sequence(zero_state):
            state = nest.pack_sequence_as(structure=zero_state,
                                                                        flat_sequence=state_list)
        return outputs_and_state[:outputs_len], state

def attention_decoder(decoder_inputs,
                        emotion,
                        imemory,
                        ememory,
                        initial_state,
                        attention_states,
                        cell,
                        output_size=None,
                        output_projection=None,
                        num_heads=1,
                        loop_function=None,
                        dtype=None,
                        scope=None,
                        initial_state_attention=True):
    """RNN decoder with attention for the sequence-to-sequence model.

    In this context "attention" means that, during decoding, the RNN can look up
    information in the additional tensor attention_states, and it does this by
    focusing on a few entries from the tensor. This model has proven to yield
    especially good results in a number of sequence-to-sequence tasks. This
    implementation is based on http://arxiv.org/abs/1412.7449 (see below for
    details). It is recommended for complex sequence-to-sequence tasks.

    Args:
        decoder_inputs: A list of 2D Tensors [batch_size x input_size].
        initial_state: 2D Tensor [batch_size x cell.state_size].
        attention_states: 3D Tensor [batch_size x attn_length x attn_size].
        cell: rnn_cell.RNNCell defining the cell function and size.
        output_size: Size of the output vectors; if None, we use cell.output_size.
        num_heads: Number of attention heads that read from attention_states.
        loop_function: If not None, this function will be applied to i-th output
            in order to generate i+1-th input, and decoder_inputs will be ignored,
            except for the first element ("GO" symbol). This can be used for decoding,
            but also for training to emulate http://arxiv.org/abs/1506.03099.
            Signature -- loop_function(prev, i) = next
                * prev is a 2D Tensor of shape [batch_size x output_size],
                * i is an integer, the step number (when advanced control is needed),
                * next is a 2D Tensor of shape [batch_size x input_size].
        dtype: The dtype to use for the RNN initial state (default: tf.float32).
        scope: VariableScope for the created subgraph; default: "attention_decoder".
        initial_state_attention: If False (default), initial attentions are zero.
            If True, initialize the attentions from the initial state and attention
            states -- useful when we wish to resume decoding from a previously
            stored decoder state and attention states.

    Returns:
        A tuple of the form (outputs, state), where:
            outputs: A list of the same length as decoder_inputs of 2D Tensors of
                shape [batch_size x output_size]. These represent the generated outputs.
                Output i is computed from input i (which is either the i-th element
                of decoder_inputs or loop_function(output {i-1}, i)) as follows.
                First, we run the cell on a combination of the input and previous
                attention masks:
                    cell_output, new_state = cell(linear(input, prev_attn), prev_state).
                Then, we calculate new attention masks:
                    new_attn = softmax(V^T * tanh(W * attention_states + U * new_state))
                and then we calculate the output:
                    output = linear(cell_output, new_attn).
            state: The state of each decoder cell the final time-step.
                It is a 2D Tensor of shape [batch_size x cell.state_size].

    Raises:
        ValueError: when num_heads is not positive, there are no inputs, shapes
            of attention_states are not set, or input size cannot be inferred
            from the input.
    """
    if not decoder_inputs:
        raise ValueError("Must provide at least 1 input to attention decoder.")
    if num_heads < 1:
        raise ValueError("With less than 1 heads, use a non-attention decoder.")
    if attention_states.get_shape()[2].value is None:
        raise ValueError("Shape[2] of attention_states must be known: %s"
                                         % attention_states.get_shape())
    if output_size is None:
        output_size = cell.output_size

    with variable_scope.variable_scope(
            scope or "attention_decoder", dtype=dtype) as scope:
        dtype = scope.dtype

        batch_size = array_ops.shape(decoder_inputs[0])[0]    # Needed for reshaping.
        attn_length = attention_states.get_shape()[1].value
        if attn_length is None:
            attn_length = shape(attention_states)[1]
        attn_size = attention_states.get_shape()[2].value

        # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
        hidden = array_ops.reshape(
                attention_states, [-1, attn_length, 1, attn_size])
        hidden_features = []
        v = []
        attention_vec_size = attn_size    # Size of query vectors for attention.
        for a in xrange(num_heads):
            k = variable_scope.get_variable("AttnW_%d" % a,
                                                                            [1, 1, attn_size, attention_vec_size])
            hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
            v.append(
                    variable_scope.get_variable("AttnV_%d" % a, [attention_vec_size]))

        state = initial_state

        def attention(query):
            """Put attention masks on hidden using hidden_features and query."""
            ds = []    # Results of attention reads will be stored here.
            if nest.is_sequence(query):    # If the query is a tuple, flatten it.
                query_list = nest.flatten(query)
                for q in query_list:    # Check that ndims == 2 if specified.
                    ndims = q.get_shape().ndims
                    if ndims:
                        assert ndims == 2
                query = array_ops.concat(1, query_list)
            for a in xrange(num_heads):
                with variable_scope.variable_scope("Attention_%d" % a):
                    y = linear(query, attention_vec_size, True)
                    y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
                    # Attention mask is a softmax of v^T * tanh(...).
                    s = math_ops.reduce_sum(
                            v[a] * math_ops.tanh(hidden_features[a] + y), [2, 3])
                    a = nn_ops.softmax(s)
                    # Now calculate the attention-weighted vector d.
                    d = math_ops.reduce_sum(
                            array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden,
                            [1, 2])
                    ds.append(array_ops.reshape(d, [-1, attn_size]))
            return ds

        outputs = []
        gates = []
        prev = None
        batch_attn_size = array_ops.pack([batch_size, attn_size])
        attns = [array_ops.zeros(batch_attn_size, dtype=dtype)
                         for _ in xrange(num_heads)]
        for a in attns:    # Ensure the second shape of attention vectors is set.
            a.set_shape([None, attn_size])
        if initial_state_attention:
            attns = attention(initial_state)
        for i, inp in enumerate(decoder_inputs):
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()
            # If loop_function is set, we use it instead of decoder_inputs.
            if loop_function is not None and prev is not None:
                with variable_scope.variable_scope("loop_function", reuse=True):
                    inp = loop_function(prev, i)
            # Merge input and previous attentions into one vector of the right size.
            input_size = inp.get_shape().with_rank(2)[1]
            if input_size.value is None:
                raise ValueError("Could not infer input size from input: %s" % inp.name)
            #x = linear([inp] + attns, input_size, True)
            x = array_ops.concat(1, [inp, attns[0]])
            # Run the RNN.
            if emotion is None and imemory is None:
                cell_output, state = cell(x, state)
            else:
                #imemory = tf.Print(imemory, [tf.reduce_sum(imemory**2)], summarize=1000)
                cell_output, state, imemory = cell(x, state, emotion, imemory)
            # Run the attention mechanism.
            if i == 0 and initial_state_attention:
                with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                                                                     reuse=True):
                    attns = attention(state)
            else:
                attns = attention(state)

            with variable_scope.variable_scope("AttnOutputProjection"):
                output = linear([cell_output] + attns, output_size, True)

            if output_projection is not None:
                logit = nn_ops.xw_plus_b(
                        output, output_projection[0], output_projection[1])

            gate = None

            if ememory is not None:

                with variable_scope.variable_scope("OutputEmemoryGate"):
                    g = tf.reshape(tf.sigmoid(linear([output], 1, True)), [-1])

                #hard gate
                #g = tf.cast(tf.greater(g, 0.5), tf.float32)
                
                output0, output1 = tf.dynamic_partition(tf.transpose(logit), tf.cast(ememory, tf.int32), 2)
                indice0, indice1 = tf.dynamic_partition(tf.range(ememory.get_shape()[-1].value), tf.cast(ememory, tf.int32), 2)
                s0 = tf.nn.softmax(output0, dim=0) * (1-g)
                s1 = tf.nn.softmax(output1, dim=0) * g
                output = tf.transpose(tf.reshape(tf.dynamic_stitch([indice0, indice1], [s0, s1]), [ememory.get_shape()[-1].value, batch_size]))
                gate = tf.stack([g, (1-g)])
            else:
                output = tf.nn.softmax(logit)

            if loop_function is not None:
                prev = output

            outputs.append(output)
            gates.append(gate)
    return outputs, state, imemory, ememory, gates

def beam_attention_decoder(decoder_inputs,
                        emotion,
                        imemory,
                        ememory,
                        initial_state,
                        attention_states,
                        cell,
                        output_size=None,
                        num_heads=1,
                        loop_function=None,
                        dtype=None,
                        scope=None,
                        initial_state_attention=True,
                        output_projection=None,
                        beam_size=10):
    """RNN decoder with attention for the sequence-to-sequence model.

    In this context "attention" means that, during decoding, the RNN can look up
    information in the additional tensor attention_states, and it does this by
    focusing on a few entries from the tensor. This model has proven to yield
    especially good results in a number of sequence-to-sequence tasks. This
    implementation is based on http://arxiv.org/abs/1412.7449 (see below for
    details). It is recommended for complex sequence-to-sequence tasks.

    Args:
        decoder_inputs: A list of 2D Tensors [batch_size x input_size].
        initial_state: 2D Tensor [batch_size x cell.state_size].
        attention_states: 3D Tensor [batch_size x attn_length x attn_size].
        cell: rnn_cell.RNNCell defining the cell function and size.
        output_size: Size of the output vectors; if None, we use cell.output_size.
        num_heads: Number of attention heads that read from attention_states.
        loop_function: If not None, this function will be applied to i-th output
            in order to generate i+1-th input, and decoder_inputs will be ignored,
            except for the first element ("GO" symbol). This can be used for decoding,
            but also for training to emulate http://arxiv.org/abs/1506.03099.
            Signature -- loop_function(prev, i) = next
                * prev is a 2D Tensor of shape [batch_size x output_size],
                * i is an integer, the step number (when advanced control is needed),
                * next is a 2D Tensor of shape [batch_size x input_size].
        dtype: The dtype to use for the RNN initial state (default: tf.float32).
        scope: VariableScope for the created subgraph; default: "attention_decoder".
        initial_state_attention: If False (default), initial attentions are zero.
            If True, initialize the attentions from the initial state and attention
            states -- useful when we wish to resume decoding from a previously
            stored decoder state and attention states.

    Returns:
        A tuple of the form (outputs, state), where:
            outputs: A list of the same length as decoder_inputs of 2D Tensors of
                shape [batch_size x output_size]. These represent the generated outputs.
                Output i is computed from input i (which is either the i-th element
                of decoder_inputs or loop_function(output {i-1}, i)) as follows.
                First, we run the cell on a combination of the input and previous
                attention masks:
                    cell_output, new_state = cell(linear(input, prev_attn), prev_state).
                Then, we calculate new attention masks:
                    new_attn = softmax(V^T * tanh(W * attention_states + U * new_state))
                and then we calculate the output:
                    output = linear(cell_output, new_attn).
            state: The state of each decoder cell the final time-step.
                It is a 2D Tensor of shape [batch_size x cell.state_size].

    Raises:
        ValueError: when num_heads is not positive, there are no inputs, shapes
            of attention_states are not set, or input size cannot be inferred
            from the input.
    """
    if not decoder_inputs:
        raise ValueError("Must provide at least 1 input to attention decoder.")
    if num_heads < 1:
        raise ValueError("With less than 1 heads, use a non-attention decoder.")
    if attention_states.get_shape()[2].value is None:
        raise ValueError("Shape[2] of attention_states must be known: %s"
                                         % attention_states.get_shape())
    if output_size is None:
        output_size = cell.output_size

    with variable_scope.variable_scope(
            scope or "attention_decoder", dtype=dtype) as scope:
        dtype = scope.dtype

        batch_size = array_ops.shape(decoder_inputs[0])[0]    # Needed for reshaping.
        attn_length = attention_states.get_shape()[1].value
        if attn_length is None:
            attn_length = shape(attention_states)[1]
        attn_size = attention_states.get_shape()[2].value

        # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
        hidden = array_ops.reshape(
                attention_states, [-1, attn_length, 1, attn_size])
        hidden_features = []
        v = []
        attention_vec_size = attn_size    # Size of query vectors for attention.
        for a in xrange(num_heads):
            k = variable_scope.get_variable("AttnW_%d" % a,
                                                                            [1, 1, attn_size, attention_vec_size])
            hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
            v.append(
                    variable_scope.get_variable("AttnV_%d" % a, [attention_vec_size]))

        state = initial_state
        state_size = int(state[0].get_shape().with_rank(2)[1])

        def attention(query):
            """Put attention masks on hidden using hidden_features and query."""
            ds = []    # Results of attention reads will be stored here.
            if nest.is_sequence(query):    # If the query is a tuple, flatten it.
                query_list = nest.flatten(query)
                for q in query_list:    # Check that ndims == 2 if specified.
                    ndims = q.get_shape().ndims
                    if ndims:
                        assert ndims == 2
                query = array_ops.concat(1, query_list)
            for a in xrange(num_heads):
                with variable_scope.variable_scope("Attention_%d" % a):
                    y = linear(query, attention_vec_size, True)
                    y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
                    # Attention mask is a softmax of v^T * tanh(...).
                    s = math_ops.reduce_sum(
                            v[a] * math_ops.tanh(hidden_features[a] + y), [2, 3])
                    a = nn_ops.softmax(s)
                    # Now calculate the attention-weighted vector d.
                    d = math_ops.reduce_sum(
                            array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden,
                            [1, 2])
                    ds.append(array_ops.reshape(d, [-1, attn_size]))
            return ds

        outputs = []
        prev = None
        batch_attn_size = array_ops.pack([batch_size, attn_size])
        attns = [array_ops.zeros(batch_attn_size, dtype=dtype)
                         for _ in xrange(num_heads)]
        for a in attns:    # Ensure the second shape of attention vectors is set.
            a.set_shape([None, attn_size])
        if initial_state_attention:
            attns = attention(initial_state)
        log_beam_probs, beam_path, beam_symbols, beam_results = [], [], [], []
        for i, inp in enumerate(decoder_inputs):
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()
            # If loop_function is set, we use it instead of decoder_inputs.
            if loop_function is not None and prev is not None:
                with variable_scope.variable_scope("loop_function", reuse=True):
                    emb = loop_function(prev, i, log_beam_probs, beam_path, beam_symbols, beam_results)
                    _state = []
                    for j in state:
                        _state.append(tf.reshape(tf.gather(j, beam_path[-1]), [-1, state_size]))
                    state = tuple(_state)
            # Merge input and previous attentions into one vector of the right size.
            input_size = inp.get_shape().with_rank(2)[1]
            if input_size.value is None:
                raise ValueError("Could not infer input size from input: %s" % inp.name)
            #x = linear([inp] + attns, input_size, True)
            if i == 0:
                x = array_ops.concat(1, [inp, attns[0]])
            else:
                x = tf.concat(1, [emb, tf.reshape(tf.gather(attns[0], beam_path[-1]), [-1, attn_size])])
            # Run the RNN.
            if emotion is None and imemory is None:
                cell_output, state = cell(x, state)
            else:
                cell_output, state, imemory = cell(x, state, emotion, imemory)
            # Run the attention mechanism.
            if i == 0 and initial_state_attention:
                with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                                                                     reuse=True):
                    attns = attention(state)
            else:
                attns = attention(state)

            with variable_scope.variable_scope("AttnOutputProjection"):
                output = linear([cell_output] + attns, output_size, True)

            if output_projection is not None:
                logit = nn_ops.xw_plus_b(
                        output, output_projection[0], output_projection[1])

            if ememory is not None:

                with variable_scope.variable_scope("OutputEmemoryGate"):
                    g = tf.reshape(tf.sigmoid(linear([output], 1, True)), [-1])

                #hard gate
                #g = tf.cast(tf.greater(g, 0.5), tf.float32)
                
                output0, output1 = tf.dynamic_partition(tf.transpose(logit), tf.cast(ememory, tf.int32), 2)
                indice0, indice1 = tf.dynamic_partition(tf.range(ememory.get_shape()[-1].value), tf.cast(ememory, tf.int32), 2)
                s0 = tf.nn.softmax(output0, dim=0) * (1-g)
                s1 = tf.nn.softmax(output1, dim=0) * g
                output = tf.transpose(tf.reshape(tf.dynamic_stitch([indice0, indice1], [s0, s1]), [ememory.get_shape()[-1].value, -1]))
            else:
                output = tf.nn.softmax(logit)

            if loop_function is not None:
                prev = output
            if i == 0:
                #emotion = tf.reshape([emotion]*beam_size, [beam_size, -1])
                if emotion is not None:
                    emotion = tf.concat(0, [emotion]*beam_size)
                if imemory is not None:
                    imemory = tf.concat(0, [imemory]*beam_size)
            outputs.append(output)
    return outputs, state, beam_results, tf.reshape(tf.concat(0, beam_symbols),[-1,beam_size]), tf.reshape(tf.concat(0, beam_path),[-1,beam_size])
    



def embedding_attention_decoder(decoder_inputs,
                                    decoder_emotions,
                                    initial_state,
                                    attention_states,
                                    cell,
                                    num_symbols,
                                    embedding_size,
                                    emotion_category,
                                    emotion_size,
                                    imemory_size,
                                    use_emb,
                                    use_imemory,
                                    use_ememory,
                                    num_heads=1,
                                    output_size=None,
                                    output_projection=None,
                                    feed_previous=False,
                                    update_embedding_for_previous=True,
                                    dtype=None,
                                    scope=None,
                                    initial_state_attention=True,
                                    beam_search=True,
                                    beam_size=10):
    """RNN decoder with embedding and attention and a pure-decoding option.

    Args:
        decoder_inputs: A list of 1D batch-sized int32 Tensors (decoder inputs).
        initial_state: 2D Tensor [batch_size x cell.state_size].
        attention_states: 3D Tensor [batch_size x attn_length x attn_size].
        cell: rnn_cell.RNNCell defining the cell function.
        num_symbols: Integer, how many symbols come into the embedding.
        embedding_size: Integer, the length of the embedding vector for each symbol.
        num_heads: Number of attention heads that read from attention_states.
        output_size: Size of the output vectors; if None, use output_size.
        output_projection: None or a pair (W, B) of output projection weights and
            biases; W has shape [output_size x num_symbols] and B has shape
            [num_symbols]; if provided and feed_previous=True, each fed previous
            output will first be multiplied by W and added B.
        feed_previous: Boolean; if True, only the first of decoder_inputs will be
            used (the "GO" symbol), and all other decoder inputs will be generated by:
                next = embedding_lookup(embedding, argmax(previous_output)),
            In effect, this implements a greedy decoder. It can also be used
            during training to emulate http://arxiv.org/abs/1506.03099.
            If False, decoder_inputs are used as given (the standard decoder case).
        update_embedding_for_previous: Boolean; if False and feed_previous=True,
            only the embedding for the first symbol of decoder_inputs (the "GO"
            symbol) will be updated by back propagation. Embeddings for the symbols
            generated from the decoder itself remain unchanged. This parameter has
            no effect if feed_previous=False.
        dtype: The dtype to use for the RNN initial states (default: tf.float32).
        scope: VariableScope for the created subgraph; defaults to
            "embedding_attention_decoder".
        initial_state_attention: If False (default), initial attentions are zero.
            If True, initialize the attentions from the initial state and attention
            states -- useful when we wish to resume decoding from a previously
            stored decoder state and attention states.

    Returns:
        A tuple of the form (outputs, state), where:
            outputs: A list of the same length as decoder_inputs of 2D Tensors with
                shape [batch_size x output_size] containing the generated outputs.
            state: The state of each decoder cell at the final time-step.
                It is a 2D Tensor of shape [batch_size x cell.state_size].

    Raises:
        ValueError: When output_projection has the wrong shape.
    """
    if output_size is None:
        output_size = cell.output_size
    if output_projection is not None:
        proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
        proj_biases.get_shape().assert_is_compatible_with([num_symbols])

    with variable_scope.variable_scope(
            scope or "embedding_attention_decoder", dtype=dtype) as scope:

        embedding = variable_scope.get_variable("embedding", [num_symbols, embedding_size])
        emotion = None
        imemory = None
        ememory = None
        if use_emb:
            emotion_embedding = variable_scope.get_variable("emotion_embedding", [emotion_category, emotion_size])
            emotion = embedding_ops.embedding_lookup(emotion_embedding, decoder_emotions)

        if use_imemory:
            internal_memory = variable_scope.get_variable("internal_memory", [emotion_category, imemory_size])
            imemory = embedding_ops.embedding_lookup(internal_memory, decoder_emotions)
            #imemory = tf.Print(imemory, [imemory, internal_memory], summarize=10)
        
        if use_ememory:
            external_memory = variable_scope.get_variable("external_memory", [emotion_category, num_symbols], trainable=False)
            ememory = embedding_ops.embedding_lookup(external_memory, decoder_emotions[0])

        if beam_search:
            loop_function = _extract_beam_search(
                embedding, beam_size,num_symbols, embedding_size, output_projection,
                update_embedding_for_previous)
        else:
            loop_function = _extract_argmax_and_embed(
                embedding, num_symbols, output_projection,
                update_embedding_for_previous) if feed_previous else None

        emb_inp = [
                embedding_ops.embedding_lookup(embedding, i) for i in decoder_inputs]
                #array_ops.concat(1, [embedding_ops.embedding_lookup(embedding, i), emotion]) for i in decoder_inputs]
        if beam_search:
            return beam_attention_decoder(
                        emb_inp, 
                        emotion, 
                        imemory,
                        ememory,
                        initial_state, 
                        attention_states, 
                        cell, 
                        output_size=output_size,
                        num_heads=num_heads, 
                        loop_function=loop_function,
                        initial_state_attention=initial_state_attention, 
                        output_projection=output_projection, 
                        beam_size=beam_size)
        else:
            return attention_decoder(
                    emb_inp,
                    emotion,
                    imemory,
                    ememory,
                    initial_state,
                    attention_states,
                    cell,
                    output_size=output_size,
                    output_projection=output_projection, 
                    num_heads=num_heads,
                    loop_function=loop_function,
                    initial_state_attention=initial_state_attention)


def embedding_attention_seq2seq(encoder_inputs,
                                    decoder_inputs,
                                    decoder_emotions,
                                    en_cell,
                                    de_cell,
                                    num_encoder_symbols,
                                    num_decoder_symbols,
                                    embedding_size,
                                    emotion_category,
                                    emotion_size,
                                    imemory_size,
                                    use_emb=False,
                                    use_imemory=False,
                                    use_ememory=False,
                                    num_heads=1,
                                    output_projection=None,
                                    feed_previous=False,
                                    dtype=None,
                                    scope=None,
                                    initial_state_attention=True,
                                    beam_search=True,
                                    beam_size=10):
    """Embedding sequence-to-sequence model with attention.

    This model first embeds encoder_inputs by a newly created embedding (of shape
    [num_encoder_symbols x input_size]). Then it runs an RNN to encode
    embedded encoder_inputs into a state vector. It keeps the outputs of this
    RNN at every step to use for attention later. Next, it embeds decoder_inputs
    by another newly created embedding (of shape [num_decoder_symbols x
    input_size]). Then it runs attention decoder, initialized with the last
    encoder state, on embedded decoder_inputs and attending to encoder outputs.

    Warning: when output_projection is None, the size of the attention vectors
    and variables will be made proportional to num_decoder_symbols, can be large.

    Args:
        encoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
        decoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
        cell: rnn_cell.RNNCell defining the cell function and size.
        num_encoder_symbols: Integer; number of symbols on the encoder side.
        num_decoder_symbols: Integer; number of symbols on the decoder side.
        embedding_size: Integer, the length of the embedding vector for each symbol.
        num_heads: Number of attention heads that read from attention_states.
        output_projection: None or a pair (W, B) of output projection weights and
            biases; W has shape [output_size x num_decoder_symbols] and B has
            shape [num_decoder_symbols]; if provided and feed_previous=True, each
            fed previous output will first be multiplied by W and added B.
        feed_previous: Boolean or scalar Boolean Tensor; if True, only the first
            of decoder_inputs will be used (the "GO" symbol), and all other decoder
            inputs will be taken from previous outputs (as in embedding_rnn_decoder).
            If False, decoder_inputs are used as given (the standard decoder case).
        dtype: The dtype of the initial RNN state (default: tf.float32).
        scope: VariableScope for the created subgraph; defaults to
            "embedding_attention_seq2seq".
        initial_state_attention: If False (default), initial attentions are zero.
            If True, initialize the attentions from the initial state and attention
            states.

    Returns:
        A tuple of the form (outputs, state), where:
            outputs: A list of the same length as decoder_inputs of 2D Tensors with
                shape [batch_size x num_decoder_symbols] containing the generated
                outputs.
            state: The state of each decoder cell at the final time-step.
                It is a 2D Tensor of shape [batch_size x cell.state_size].
    """
    with variable_scope.variable_scope(
            scope or "embedding_attention_seq2seq", dtype=dtype) as scope:
        dtype = scope.dtype
        # Encoder.
        encoder_cell = rnn_cell.EmbeddingWrapper(
                en_cell, embedding_classes=num_encoder_symbols,
                embedding_size=embedding_size)
        encoder_outputs, encoder_state = rnn.rnn(
                encoder_cell, encoder_inputs, dtype=dtype)

        # First calculate a concatenation of encoder outputs to put attention on.
        top_states = [array_ops.reshape(e, [-1, 1, en_cell.output_size])
                                    for e in encoder_outputs]
        attention_states = array_ops.concat(1, top_states)

        # Decoder.
        output_size = None
        if output_projection is None:
            #not work
            de_cell = rnn_cell.OutputProjectionWrapper(de_cell, num_decoder_symbols)
            output_size = num_decoder_symbols

        if isinstance(feed_previous, bool):
            return embedding_attention_decoder(
                    decoder_inputs,
                    decoder_emotions,
                    encoder_state,
                    attention_states,
                    de_cell,
                    num_decoder_symbols,
                    embedding_size,
                    emotion_category=emotion_category,
                    emotion_size=emotion_size,
                    imemory_size=imemory_size,
                    use_emb=use_emb,
                    use_imemory=use_imemory,
                    use_ememory=use_ememory,
                    num_heads=num_heads,
                    output_size=output_size,
                    output_projection=output_projection,
                    feed_previous=feed_previous,
                    initial_state_attention=initial_state_attention,
                    beam_search=beam_search,
                    beam_size=beam_size)



def sequence_loss_by_example(logits, targets, weights, ememory,
                                                         average_across_timesteps=True,
                                                         softmax_loss_function=None, name=None):
    """Weighted cross-entropy loss for a sequence of logits (per example).

    Args:
        logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
        targets: List of 1D batch-sized int32 Tensors of the same length as logits.
        weights: List of 1D batch-sized float-Tensors of the same length as logits.
        average_across_timesteps: If set, divide the returned cost by the total
            label weight.
        softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
            to be used instead of the standard softmax (the default if this is None).
        name: Optional name for this operation, default: "sequence_loss_by_example".

    Returns:
        1D batch-sized float Tensor: The log-perplexity for each sequence.

    Raises:
        ValueError: If len(logits) is different from len(targets) or len(weights).
    """
    if len(targets) != len(logits) or len(weights) != len(logits):
        raise ValueError("Lengths of logits, weights, and targets must be the same "
                                         "%d, %d, %d." % (len(logits), len(weights), len(targets)))
    with ops.name_scope(name, "sequence_loss_by_example",
                                            logits + targets + weights if ememory is None else logits + targets + weights + [ememory]):
        log_perp_list = []
        for logit, target, weight in zip(logits, targets, weights):
            if softmax_loss_function is None:
                # TODO(irving,ebrevdo): This reshape is needed because
                # sequence_loss_by_example is called with scalars sometimes, which
                # violates our general scalar strictness policy.
                #target = array_ops.reshape(target, [-1])
                #crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
                #        logit, target)
                if ememory is None:
                    target = array_ops.reshape(target, [-1])
                    label = tf.one_hot(target, depth=logit.get_shape().with_rank(2)[1], dtype=tf.float32)
                    crossent = -tf.reduce_sum(label * tf.log(logit+1e-12), 1)
                else:
                    golden = tf.gather(ememory, target)
                    golden = tf.stack([golden, 1-golden])
                    crossent = -tf.reduce_sum(golden * tf.log(logit+1e-12), 0)

            else:
                #sampled softmax not work
                crossent = softmax_loss_function(logit, target)
            log_perp_list.append(crossent * weight)
        log_perps = math_ops.add_n(log_perp_list)
        if average_across_timesteps:
            total_size = math_ops.add_n(weights)
            total_size += 1e-12    # Just to avoid division by 0 for all-0 weights.
            log_perps /= total_size
    return log_perps


def sequence_loss(logits, targets, weights, ememory,
                                    average_across_timesteps=True, average_across_batch=True,
                                    softmax_loss_function=None, name=None):
    """Weighted cross-entropy loss for a sequence of logits, batch-collapsed.

    Args:
        logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
        targets: List of 1D batch-sized int32 Tensors of the same length as logits.
        weights: List of 1D batch-sized float-Tensors of the same length as logits.
        average_across_timesteps: If set, divide the returned cost by the total
            label weight.
        average_across_batch: If set, divide the returned cost by the batch size.
        softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
            to be used instead of the standard softmax (the default if this is None).
        name: Optional name for this operation, defaults to "sequence_loss".

    Returns:
        A scalar float Tensor: The average log-perplexity per symbol (weighted).

    Raises:
        ValueError: If len(logits) is different from len(targets) or len(weights).
    """
    with ops.name_scope(name, "sequence_loss", logits + targets + weights if ememory is None else logits + targets + weights + [ememory]):
        p = sequence_loss_by_example(
                logits, targets, weights, ememory,
                average_across_timesteps=average_across_timesteps,
                softmax_loss_function=softmax_loss_function)
        cost_p = math_ops.reduce_sum(p)
        if average_across_batch:
            batch_size = array_ops.shape(targets[0])[0]
            return cost_p / math_ops.cast(batch_size, cost_p.dtype)
        else:
            return cost_p


def model_with_buckets(encoder_inputs, decoder_inputs, targets, weights, decoder_emotions,
                                             buckets, seq2seq, softmax_loss_function=None,
                                             per_example_loss=False, use_imemory=False, use_ememory=False, name=None):
    """Create a sequence-to-sequence model with support for bucketing.

    The seq2seq argument is a function that defines a sequence-to-sequence model,
    e.g., seq2seq = lambda x, y: basic_rnn_seq2seq(x, y, rnn_cell.GRUCell(24))

    Args:
        encoder_inputs: A list of Tensors to feed the encoder; first seq2seq input.
        decoder_inputs: A list of Tensors to feed the decoder; second seq2seq input.
        targets: A list of 1D batch-sized int32 Tensors (desired output sequence).
        weights: List of 1D batch-sized float-Tensors to weight the targets.
        buckets: A list of pairs of (input size, output size) for each bucket.
        seq2seq: A sequence-to-sequence model function; it takes 2 input that
            agree with encoder_inputs and decoder_inputs, and returns a pair
            consisting of outputs and states (as, e.g., basic_rnn_seq2seq).
        softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
            to be used instead of the standard softmax (the default if this is None).
        per_example_loss: Boolean. If set, the returned loss will be a batch-sized
            tensor of losses for each sequence in the batch. If unset, it will be
            a scalar with the averaged loss from all examples.
        name: Optional name for this operation, defaults to "model_with_buckets".

    Returns:
        A tuple of the form (outputs, losses), where:
            outputs: The outputs for each bucket. Its j'th element consists of a list
                of 2D Tensors. The shape of output tensors can be either
                [batch_size x output_size] or [batch_size x num_decoder_symbols]
                depending on the seq2seq model used.
            losses: List of scalar Tensors, representing losses for each bucket, or,
                if per_example_loss is set, a list of 1D batch-sized float Tensors.

    Raises:
        ValueError: If length of encoder_inputsut, targets, or weights is smaller
            than the largest (last) bucket.
    """
    if len(encoder_inputs) < buckets[-1][0]:
        raise ValueError("Length of encoder_inputs (%d) must be at least that of la"
                                         "st bucket (%d)." % (len(encoder_inputs), buckets[-1][0]))
    if len(targets) < buckets[-1][1]:
        raise ValueError("Length of targets (%d) must be at least that of last"
                                         "bucket (%d)." % (len(targets), buckets[-1][1]))
    if len(weights) < buckets[-1][1]:
        raise ValueError("Length of weights (%d) must be at least that of last"
                                         "bucket (%d)." % (len(weights), buckets[-1][1]))

    all_inputs = encoder_inputs + decoder_inputs + targets + weights + [decoder_emotions]
    losses = []
    outputs = []
    ppxes = []
    with ops.name_scope(name, "model_with_buckets", all_inputs):
        for j, bucket in enumerate(buckets):
            with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                                                                 reuse=True if j > 0 else None):
                bucket_outputs, _, imemory, ememory, gates = seq2seq(encoder_inputs[:bucket[0]],
                                                                        decoder_inputs[:bucket[1]], decoder_emotions)
                outputs.append(bucket_outputs)
                if per_example_loss:
                    #not work
                    losses.append(math_ops.reduce_sum(imemory**2)/math_ops.cast(array_ops.shape(targets[0])[0], imemory.dtype)+sequence_loss_by_example(
                            outputs[-1], targets[:bucket[1]], weights[:bucket[1]],
                            softmax_loss_function=softmax_loss_function))
                else:
                    loss = sequence_loss(
                            outputs[-1], targets[:bucket[1]], weights[:bucket[1]], None,
                            softmax_loss_function=softmax_loss_function)
                    ppxes.append(loss)
                    if use_imemory:
                        loss += math_ops.reduce_sum(imemory**2)/math_ops.cast(array_ops.shape(targets[0])[0], imemory.dtype) 
                    if use_ememory:
                        loss += sequence_loss(
                            gates, targets[:bucket[1]], weights[:bucket[1]], ememory,
                            softmax_loss_function=softmax_loss_function)
                    losses.append(loss)

    return outputs, losses, ppxes

def decode_model_with_buckets(encoder_inputs, decoder_inputs, targets, weights, decoder_emotions,
                                             buckets, seq2seq, softmax_loss_function=None,
                                             per_example_loss=False, name=None):
    """Create a sequence-to-sequence model with support for bucketing.
    The seq2seq argument is a function that defines a sequence-to-sequence model,
    e.g., seq2seq = lambda x, y: basic_rnn_seq2seq(x, y, rnn_cell.GRUCell(24))
    Args:
        encoder_inputs: A list of Tensors to feed the encoder; first seq2seq input.
        decoder_inputs: A list of Tensors to feed the decoder; second seq2seq input.
        targets: A list of 1D batch-sized int32 Tensors (desired output sequence).
        weights: List of 1D batch-sized float-Tensors to weight the targets.
        buckets: A list of pairs of (input size, output size) for each bucket.
        seq2seq: A sequence-to-sequence model function; it takes 2 input that
            agree with encoder_inputs and decoder_inputs, and returns a pair
            consisting of outputs and states (as, e.g., basic_rnn_seq2seq).
        softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
            to be used instead of the standard softmax (the default if this is None).
        per_example_loss: Boolean. If set, the returned loss will be a batch-sized
            tensor of losses for each sequence in the batch. If unset, it will be
            a scalar with the averaged loss from all examples.
        name: Optional name for this operation, defaults to "model_with_buckets".
    Returns:
        A tuple of the form (outputs, losses), where:
            outputs: The outputs for each bucket. Its j'th element consists of a list
                of 2D Tensors of shape [batch_size x num_decoder_symbols] (jth outputs).
            losses: List of scalar Tensors, representing losses for each bucket, or,
                if per_example_loss is set, a list of 1D batch-sized float Tensors.
    Raises:
        ValueError: If length of encoder_inputsut, targets, or weights is smaller
            than the largest (last) bucket.
    """
    if len(encoder_inputs) < buckets[-1][0]:
        raise ValueError("Length of encoder_inputs (%d) must be at least that of la"
                                         "st bucket (%d)." % (len(encoder_inputs), buckets[-1][0]))
    if len(targets) < buckets[-1][1]:
        raise ValueError("Length of targets (%d) must be at least that of last"
                                         "bucket (%d)." % (len(targets), buckets[-1][1]))
    if len(weights) < buckets[-1][1]:
        raise ValueError("Length of weights (%d) must be at least that of last"
                                         "bucket (%d)." % (len(weights), buckets[-1][1]))

    all_inputs = encoder_inputs + decoder_inputs + targets + weights + [decoder_emotions]

    losses = []
    outputs = []
    beam_results = []
    beam_symbols = []
    beam_parents = []
    with ops.name_scope(name, "model_with_buckets", all_inputs):
        for j, bucket in enumerate(buckets):
            with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                                                                 reuse=True if j > 0 else None):
                bucket_outputs, _, beam_result, beam_symbol, beam_parent = seq2seq(encoder_inputs[:bucket[0]],
                                                                        decoder_inputs[:bucket[1]], decoder_emotions)
                outputs.append(bucket_outputs)
                beam_results.append(beam_result)
                beam_symbols.append(beam_symbol)
                beam_parents.append(beam_parent)
    print("End**********")

    return outputs, beam_results, beam_symbols, beam_parents
