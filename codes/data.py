from fastNLP import DataSet, Instance, Vocabulary
import os


class Data:
    """Train, valid, test datasets and corresponding dictionaries."""
    def __init__(self, file_path, vocab_min_freq=-1):
        self.word_vocab, self.item_vocab, self.context_vocab = \
            self.construct_vocab(
                os.path.join(file_path, 'train.txt'), vocab_min_freq)

        self.train = self.tokenize(os.path.join(file_path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(file_path, 'val.txt'))
        self.test = self.tokenize(os.path.join(file_path, 'test.txt'))

        self.output_length = max([len(x['output']) for x in self.train])

    @staticmethod
    def get_tag(tokens, tag):
        """Extracts the value inside the given tag."""
        return tokens[tokens.index('<' + tag + '>') +
                      1:tokens.index('</' + tag + '>')]

    def construct_vocab(self, file_name, vocab_min_freq):
        word_vocab = Vocabulary(min_freq=vocab_min_freq)
        item_vocab = Vocabulary()
        context_vocab = Vocabulary()
        with open(file_name) as fd:
            for line in fd:
                sen_tokens = line.strip().split()
                word_vocab.add_word_lst(self.get_tag(sen_tokens, 'dialogue'))
                item_vocab.add_word_lst(self.get_tag(sen_tokens, 'output'))
                context_vocab.add_word_lst(self.get_tag(sen_tokens, 'input'))
        word_vocab.build_vocab()
        item_vocab.build_vocab()
        context_vocab.build_vocab()
        return word_vocab, item_vocab, context_vocab

    def tokenize(self, file_name):
        """Tokenizes the file and produces a dataset."""
        dataset = DataSet()
        with open(file_name) as fd:
            for line in fd:
                tokens = line.split()
                input_index = [self.context_vocab[word]
                               for word in self.get_tag(tokens, 'input')]
                word_index = [self.word_vocab[word]
                              for word in self.get_tag(tokens, 'dialogue')]
                item_index = [self.item_vocab[word]
                              for word in self.get_tag(tokens, 'output')]
                dataset.append(Instance(input=input_index,
                               dialogue=word_index,
                               output=item_index))
        for name in ['input', 'dialogue', 'output']:
            dataset.set_input(name)
        return dataset
