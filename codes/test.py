from data import Data
from dialog_model import DialogModel
from fastNLP import Batch
from fastNLP.core.sampler import SequentialSampler
from train import Criterion, Trainer
import torch


batch_size = 25


with open('../model/10-20.0', 'rb') as f:
    model = torch.load(f)

dataset = Data('../data', 20)

test_batch = Batch(dataset.test, batch_size, SequentialSampler())

trainer = Trainer(model)
test_word_loss, test_select_loss = 0, 0
for batch, _ in test_batch:
    input_context, output_item, dialogue_input, dialogue_target = \
        trainer.split_batch_data(batch)

    (dialogue_output, dialogue_language_model_out, dialogue_target,
     selection_decoder_outs, output_item) = model.forward(
        input_context, dialogue_input, dialogue_target, output_item)

    test_word_loss += trainer.word_criterion(
        dialogue_output.view(-1, len(model.word_dict)),
        dialogue_target).item()
    test_select_loss += trainer.selection_criterion(
        selection_decoder_outs, output_item).item()

test_word_loss /= len(test_batch.dataset)
test_select_loss /= len(test_batch.dataset)
print('test_word_loss %.3f, test_select_loss %.3f' %
      (test_word_loss, test_select_loss))
