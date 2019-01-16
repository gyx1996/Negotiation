from data import Data


class DataTest:
    def __init__(self):
        self.test()

    def test(self):
        print('Testing Data Module...')
        dataset = Data('../data')
        assert dataset.output_length == 370
        print('Pass!')
        print(dataset.word_vocab['me'])


DataTest()
