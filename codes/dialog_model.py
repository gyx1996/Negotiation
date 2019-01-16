from domain import ObjectDivisionDomain
from fastNLP.modules.encoder.embedding import Embedding
from fastNLP.modules.encoder.masked_rnn import MaskedGRU
from fastNLP.modules.encoder.linear import Linear
from fastNLP.modules.encoder.variational_rnn import VarGRU
from torch.nn.functional import linear, softmax
from torch import nn
import torch
word_embedding_dim = 256
context_embedding_dim = 64
context_hidden_num = 64
attention_hidden_num = 64
selection_hidden_num = 64
language_model_hidden_num = 256
dropout = 0.5


class RnnContextEncoder(nn.Module):
    """A module that encodes dialogues context using an RNN."""
    def __init__(self, n, embed_dim, hidden_num):
        super().__init__()
        self.context_embedding = Embedding(n, embed_dim)
        self.context_encoder = MaskedGRU(embed_dim, hidden_num)

    def forward(self, ctx):
        _, ctx_h = self.context_encoder(self.context_embedding(ctx))
        return ctx_h


class DialogModel(nn.Module):
    def __init__(self, word_dict, item_dict, context_dict, output_length):
        super(DialogModel, self).__init__()
        domain = ObjectDivisionDomain()

        # vocabulary:
        self.word_dict = word_dict
        self.item_dict = item_dict
        self.context_dict = context_dict

        # word embedding:
        self.word_encoder = nn.Embedding(
            len(self.word_dict),
            word_embedding_dim)

        self.dropout = nn.Dropout(dropout)

        # context:
        self.context_encoder = RnnContextEncoder(
            len(self.context_dict),
            context_embedding_dim,
            context_hidden_num)

        # language model:
        self.language_model_reader = nn.GRU(
            context_hidden_num + word_embedding_dim,
            language_model_hidden_num)
        self.language_model_decoder = nn.Linear(
            language_model_hidden_num, word_embedding_dim)
        self.language_model_writer = nn.GRUCell(
            context_hidden_num + word_embedding_dim,
            language_model_hidden_num)
        self.language_model_writer.weight_ih = \
            self.language_model_reader.weight_ih_l0
        self.language_model_writer.weight_hh =\
            self.language_model_reader.weight_hh_l0
        self.language_model_writer.bias_ih = \
            self.language_model_reader.bias_ih_l0
        self.language_model_writer.bias_hh = \
            self.language_model_reader.bias_hh_l0

        # selection:
        self.selection_bi_rnn = MaskedGRU(
            language_model_hidden_num + word_embedding_dim,
            attention_hidden_num,
            bidirectional=True)
        self.selection_attention = nn.Sequential(
            Linear(2 * attention_hidden_num, attention_hidden_num),
            nn.Tanh(), Linear(attention_hidden_num, 1))
        self.selection_encoder = nn.Sequential(
            Linear(2 * attention_hidden_num + context_hidden_num,
                   selection_hidden_num), nn.Tanh())
        self.selection_decoders = nn.ModuleList()
        for i in range(output_length):
            self.selection_decoders.append(
                Linear(selection_hidden_num, len(self.item_dict)))

        # special token mask
        self.special_token_mask = torch.FloatTensor(len(self.word_dict))
        for i in range(len(self.word_dict)):
            w = self.word_dict.to_word(i)
            special = domain.item_pattern.match(w) or w in (
                '<unk>', 'YOU:', 'THEM:', '<pad>')
            self.special_token_mask[i] = -999 if special else 0.0

    def forward_context(self, input_context):
        return self.context_encoder(input_context)

    def forward_language_model(self, dialogue_input, context_hidden):
        dialogue_input_embedded = self.word_encoder(dialogue_input)
        context_hidden_expanded = context_hidden.expand(
            dialogue_input.size(0),
            context_hidden.size(1),
            context_hidden.size(2))
        dialogue_input_embedded = torch.cat(
            [dialogue_input_embedded, context_hidden_expanded], 2)
        dialogue_input_embedded = self.dropout(dialogue_input_embedded)

        self.language_model_reader.flatten_parameters()
        dialogue_language_model_out, _ = self.language_model_reader(
            dialogue_input_embedded)
        dialogue_embedding_out = self.language_model_decoder(
            dialogue_language_model_out.view(
                -1, dialogue_language_model_out.size(2)))
        dialogue_output = linear(dialogue_embedding_out,
                                 self.word_encoder.weight)
        dialogue_output = dialogue_output.view(
            dialogue_language_model_out.size(0),
            dialogue_language_model_out.size(1),
            dialogue_output.size(1))

        # dialogue_output: [dialogue_length, batch_size, word_num]
        # dialogue_language_model_out:
        #   [dialogue_length, batch_size, language_model_hidden_num]
        return dialogue_output, dialogue_language_model_out

    def forward_selection(self, dialogue_input, context_hidden,
                          dialogue_language_model_out):
        dialogue_input_embedded = self.word_encoder(dialogue_input)
        # selection_hidden: [dialogue_length, batch_size,
        #   language_model_hidden_num + word_embedding_dim]
        selection_hidden = torch.cat(
            [dialogue_language_model_out, dialogue_input_embedded], 2)
        selection_hidden = self.dropout(selection_hidden)
        # selection_hidden:
        #   [dialogue_length, batch_size, attention_hidden_num * 2]
        selection_hidden, _ = self.selection_bi_rnn(selection_hidden)
        # selection_hidden:
        #   [batch_size, dialogue_length, attention_hidden_num * 2]
        selection_hidden = selection_hidden.transpose(0, 1).contiguous()
        # selection_attention_logits: [batch_size, dialogue_length]
        selection_attention_logits = self.selection_attention(
            selection_hidden.view(-1, selection_hidden.size(2))).view(
            selection_hidden.size(0), selection_hidden.size(1))
        # selection_attention_prob:
        #   [dialogue_length, batch_size, attention_hidden_num * 2]
        selection_attention_prob = softmax(
            selection_attention_logits,
            dim=1).unsqueeze(2).expand_as(selection_hidden)
        # attention_hidden:
        #   [1, batch_size, attention_hidden_num * 2]
        attention_hidden = torch.sum(
            torch.mul(selection_hidden, selection_attention_prob), 1,
            keepdim=True).transpose(0, 1).contiguous()
        # h: [batch_size, attention_hidden_num * 2 + context_hidden_num]
        h = torch.cat([attention_hidden, context_hidden], 2).squeeze(0)
        h = self.dropout(h)
        # selection_hidden: [batch_size, selection_hidden_num]
        selection_hidden = self.selection_encoder.forward(h)
        selection_decoder_outs = [
            decoder.forward(selection_hidden)
            for decoder in self.selection_decoders]
        # selection_decoder_outs: [[batch_size, len(self.item_dict)]]
        # len(selection_decoder_outs) = output_length
        # return: [output_length * batch_size, len(self.item_dict)]
        return torch.cat(selection_decoder_outs)

    def forward(self,
                input_context,
                dialogue_input,
                dialogue_target,
                output_item):
        context_hidden = self.forward_context(input_context)
        dialogue_output, dialogue_language_model_out = \
            self.forward_language_model(
                dialogue_input, context_hidden)
        selection_decoder_outs = self.forward_selection(
            dialogue_input, context_hidden,
            dialogue_language_model_out)
        return (dialogue_output,
                dialogue_language_model_out,
                dialogue_target,
                selection_decoder_outs,
                output_item)
