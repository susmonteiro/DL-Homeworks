import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


def reshape_state(state):
    h_state = state[0]
    c_state = state[1]
    new_h_state = torch.cat([h_state[:-1], h_state[1:]], dim=2)
    new_c_state = torch.cat([c_state[:-1], c_state[1:]], dim=2)
    return (new_h_state, new_c_state)


class Attention(nn.Module):
    def __init__(
        self,
        hidden_size,
    ):

        super(Attention, self).__init__()
        "Luong et al. general attention (https://arxiv.org/pdf/1508.04025.pdf)"
        self.linear_in = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear_out = nn.Linear(hidden_size * 2, hidden_size)   # 256, 128

    def forward(
        self,
        query,
        encoder_outputs,
        src_lengths,
    ):
        # query: (batch_size, 1, hidden_dim)
        # encoder_outputs: (batch_size, max_src_len, hidden_dim)
        # src_lengths: (batch_size)
        # we will need to use this mask to assign float("-inf") in the attention scores
        # of the padding tokens (such that the output of the softmax is 0 in those positions)
        # Tip: use torch.masked_fill to do this
        # src_seq_mask: (batch_size, max_src_len)
        # the "~" is the elementwise NOT operator
        src_seq_mask = ~self.sequence_mask(src_lengths)
        z = self.linear_in(query)
        scores = torch.bmm(z, encoder_outputs.transpose(1, 2))
        scores = torch.masked_fill(scores, src_seq_mask.unsqueeze(1), float("-inf"))
        alignment = torch.softmax(scores, 2)
        context = torch.bmm(alignment, encoder_outputs)
        q_c = torch.cat([query, context], dim=2)
        output = self.linear_out(q_c)
        attn_out = torch.tanh(output)

        # Hints:
        # - Use torch.bmm to do the batch matrix multiplication
        #    (it does matrix multiplication for each sample in the batch)
        # - Use torch.softmax to do the softmax
        # - Use torch.tanh to do the tanh
        # - Use torch.masked_fill to do the masking of the padding tokens
        # attn_out: (batch_size, 1, hidden_size)

        return attn_out

    def sequence_mask(self, lengths):
        """
        Creates a boolean mask from sequence lengths.
        """
        batch_size = lengths.numel()
        max_len = lengths.max()
        return (
            torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1))
        )


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        hidden_size,
        padding_idx,
        dropout,
    ):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size // 2
        self.dropout = dropout

        self.embedding = nn.Embedding(
            src_vocab_size,
            hidden_size,
            padding_idx=padding_idx,
        )
        self.lstm = nn.LSTM(
            hidden_size,
            self.hidden_size,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(self.dropout)

    def forward(
        self,
        src,
        lengths,
    ):
        # src: (batch_size, max_src_len)
        # lengths: (batch_size)

        # Embed the source sequence
        embedded = self.embedding(src)

        # TODO dropout here?
        embedded_dropout = self.dropout(embedded)
        # Pack the padded sequences (before passing them to the LSTM)
        # lengths, indices = lengths.sort(descending=True)
        # src = src[indices]
        packed_src = pack(embedded_dropout, lengths, batch_first=True, enforce_sorted=False)

        packed_output, final_hidden = self.lstm(packed_src)

        # Unpack the packed sequence (after passing them to the LSTM)  
        enc_output, _ = unpack(packed_output, batch_first=True)
        enc_output = self.dropout(enc_output)


        final_hidden = self._reshape_hidden(final_hidden)

        # enc_output: (batch_size, max_src_len, hidden_size)
        # final_hidden: tuple with 2 tensors
        # each tensor is (num_layers * num_directions, batch_size, hidden_size)
        return enc_output, final_hidden

    def _merge_tensor(self, state_tensor):
        forward_states = state_tensor[::2]
        backward_states = state_tensor[1::2]
        return torch.cat([forward_states, backward_states], 2)

    def _reshape_hidden(self, hidden):
        """
        hidden:
            num_layers * num_directions x batch x self.hidden_size // 2
            or a tuple of these
        returns:
            num_layers
        """
        if isinstance(hidden, tuple):
            return tuple(self._merge_tensor(h) for h in hidden)
        else:
            return self._merge_tensor(hidden)


class Decoder(nn.Module):
    def __init__(
        self,
        hidden_size,
        tgt_vocab_size,
        attn,
        padding_idx,
        dropout,
    ):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.tgt_vocab_size = tgt_vocab_size
        self.dropout = dropout

        self.embedding = nn.Embedding(
            self.tgt_vocab_size, self.hidden_size, padding_idx=padding_idx
        )

        self.dropout = nn.Dropout(self.dropout)
        self.lstm = nn.LSTM(
            self.hidden_size,
            self.hidden_size,
            batch_first=True,
        )

        self.attn = attn

    def forward(
        self,
        tgt,                # (batch_size, max_tgt_len)
        dec_state,          # should be the final hidden state from the Encoder
        encoder_outputs,    # (batch_size, max_src_len, hidden_size)
        src_lengths,        # (batch_size)
    ):

        # dec_state: tuple with 2 tensors
        # each tensor is (num_layers * num_directions, batch_size, hidden_size)

        # bidirectional encoder outputs are concatenated, so we may need to
        # reshape the decoder states to be of size (num_layers, batch_size, 2*hidden_size)
        # if they are of size (num_layers*num_directions, batch_size, hidden_size)        

        # initially each tensor has dimensions [2, 64, 64]
        # after reshaping, each tensor has dimensions [1, 64, 128]

        if dec_state[0].shape[0] == 2:
            dec_state = reshape_state(dec_state)

        tgt_old = tgt.shape # TODO delete me
        if (tgt.size(1) > 1):
            tgt = tgt[:, :-1] 
     
        # Embed the target sequence
        embedded = self.embedding(tgt)    
        embedded_dropout = self.dropout(embedded)

        outputs = []

        for t in torch.split(embedded_dropout, 1, dim=1):
            output, dec_state = self.lstm(t, dec_state)

            if self.attn is not None:
                output = self.attn(
                    output,
                    encoder_outputs,
                    src_lengths,
                )
            
            output = self.dropout(output)
            outputs.append(output)


        outputs = torch.cat(outputs, dim=1)
        return outputs, dec_state

class Seq2Seq(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
    ):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.generator = nn.Linear(decoder.hidden_size, decoder.tgt_vocab_size)

        self.generator.weight = self.decoder.embedding.weight

    def forward(
        self,
        src,
        src_lengths,
        tgt,
        dec_hidden=None,
    ):

        encoder_outputs, final_enc_state = self.encoder(src, src_lengths)

        if dec_hidden is None:
            dec_hidden = final_enc_state

        output, dec_hidden = self.decoder(
            tgt, dec_hidden, encoder_outputs, src_lengths
        )

        return self.generator(output), dec_hidden
