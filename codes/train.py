from data import Data
from dialog_model import DialogModel
from fastNLP import Batch
from fastNLP.core.sampler import SequentialSampler
import torch
from torch import nn
from torch import optim
import copy


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 25
selection_weight = 1.0
lr = 20.0
momentum = 0.0
nesterov = False
clip = 0.2
max_epoch = 30
min_learning_rate = 1e-5
decay_rate = 9.0
decay_every = 1


class Criterion(object):
    """Weighted CrossEntropyLoss."""
    def __init__(self, vocabulary, bad_tokens=None):
        if bad_tokens is None:
            bad_tokens = []
        w = torch.Tensor(len(vocabulary)).fill_(1)
        for token in bad_tokens:
            w[vocabulary[token]] = 0.0
        self.criterion = nn.CrossEntropyLoss(
            w, reduction='mean')

    def __call__(self, out, tgt):
        return self.criterion(out, tgt)


class Trainer:
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            nesterov=(nesterov and momentum > 0))
        self.word_criterion = Criterion(model.word_dict)
        self.selection_criterion = Criterion(
            model.item_dict, bad_tokens=['<disconnect>', '<disagree>'])

    @staticmethod
    def split_batch_data(batch):
        # input_context: [input_context_length(6), batch_size]
        input_context = batch['input'].transpose(0, 1).contiguous()
        # dialogue: [dialogue_length, batch_size]
        dialogue = batch['dialogue'].transpose(0, 1).contiguous()
        # output_item: [output_item_length(6) * batch_size]
        output_item = batch['output'].transpose(0, 1).contiguous().view(-1)

        # dialogue_input: [dialogue_length - 1, batch_size]
        dialogue_input = dialogue.narrow(0, 0, dialogue.size(0) - 1)
        # dialogue_target: [(dialogue_length - 1) * batch_size]
        dialogue_target = dialogue.narrow(0, 1, dialogue.size(0) - 1).view(-1)
        return input_context, output_item, dialogue_input, dialogue_target

    def iteration(self, epoch, current_learning_rate,
                  train_batch, valid_batch):
        self.model.train()
        train_total_loss = 0.0
        for batch, _ in train_batch:
            input_context, output_item, dialogue_input, dialogue_target = \
                self.split_batch_data(batch)

            input_context.to(device)
            output_item.to(device)
            dialogue_input.to(device)
            dialogue_target.to(device)

            (dialogue_output, dialogue_language_model_out, dialogue_target,
             selection_decoder_outs, output_item) = self.model.forward(
                input_context, dialogue_input, dialogue_target, output_item)

            language_model_loss = self.word_criterion(
                dialogue_output.view(-1, len(self.model.word_dict)), dialogue_target)
            selection_loss = self.selection_criterion(
                selection_decoder_outs, output_item)
            loss = language_model_loss + selection_loss * selection_weight
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            self.optimizer.step()

            train_total_loss += loss.item()

        train_loss = train_total_loss / len(train_batch.dataset)
        print('epoch %03d: train_loss %.3f, lr %0.8f' %
              (epoch, train_loss, current_learning_rate))

        valid_word_loss, valid_select_loss = 0, 0
        for batch, _ in valid_batch:
            input_context, output_item, dialogue_input, dialogue_target = \
                self.split_batch_data(batch)

            (dialogue_output, dialogue_language_model_out, dialogue_target,
             selection_decoder_outs, output_item) = self.model.forward(
                input_context, dialogue_input, dialogue_target, output_item)

            valid_word_loss += self.word_criterion(
                dialogue_output.view(-1, len(self.model.word_dict)),
                dialogue_target).item()
            valid_select_loss += self.selection_criterion(
                selection_decoder_outs, output_item).item()

        valid_word_loss /= len(valid_batch.dataset)
        valid_select_loss /= len(valid_batch.dataset)
        print('epoch %03d: valid_word_loss %.3f, valid_select_loss %.3f' %
              (epoch, valid_word_loss, valid_select_loss))
        return train_loss, valid_word_loss, valid_select_loss

    def train(self, dataset, learning_rate):
        best_model, best_valid_select_loss = None, 1e20
        last_decay_epoch = 0

        train_batch = Batch(dataset.train, batch_size, SequentialSampler())
        valid_batch = Batch(dataset.valid, batch_size, SequentialSampler())
        for epoch in range(1, max_epoch + 1):
            train_loss, valid_word_loss, valid_select_loss = self.iteration(
                epoch, learning_rate, train_batch, valid_batch)
            if valid_select_loss < best_valid_select_loss:
                best_valid_select_loss = valid_select_loss
                best_model = copy.deepcopy(self.model)

            if not epoch % 10:
                with open('../model/' + str(epoch) + '-' + str(learning_rate),
                          'wb') as f:
                    torch.save(self.model, f)

        print('Anneal: best valid select loss %.3f' % best_valid_select_loss)
        with open('../model/best-before-' + str(max_epoch), 'wb') as f:
            torch.save(self.model, f)

        self.model = best_model
        for epoch in range(max_epoch + 1, 100):
            if epoch - last_decay_epoch >= decay_every:
                last_decay_epoch = epoch
                learning_rate /= decay_rate
                if learning_rate < min_learning_rate:
                    break
                self.optimizer = optim.SGD(self.model.parameters(),
                                           lr=learning_rate)
            train_loss, valid_word_loss, valid_select_loss = self.iteration(
                epoch, learning_rate, train_batch, valid_batch)
            if not epoch % 10:
                with open('../model/' + str(epoch) + '-' + str(learning_rate),
                          'wb') as f:
                    torch.save(self.model, f)


def main():
    print('Start Training...')

    print('Loading Dataset...')
    dataset = Data('../data/negotiate', 20)
    print('Dataset Loaded!')

    print('Building model...')
    model = DialogModel(dataset.word_vocab,
                        dataset.item_vocab,
                        dataset.context_vocab,
                        dataset.output_length)
    model.to(device)
    print('Model Built!')

    print('Training...')
    trainer = Trainer(model)
    trainer.train(dataset, lr)
    print('Training Done!')


if __name__ == '__main__':
    main()
