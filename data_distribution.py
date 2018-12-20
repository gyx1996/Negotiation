from collections import OrderedDict
import numpy as np


STOP_WORDS = {'YOU:', 'THEM:', 'and', 'i', 'you', 'have', 'would', 'like',
              '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
              'hat', 'hats', 'ball', 'balls', 'book', 'books',
              '<eos>', '<selection>', '.', 'the', 'can', 'deal', ',',
              "i'll", 'to', 'a', 'an', 'for', 'get', 'take', 'will', 'ok',
              "i'd", 'no', 'me', 'one', '?', 'give', 'need', 'that', 'all',
              'if', 'two', 'how', 'about', 'of', 'okay', 'rest', 'want', '!',
              'else', 'everything', 'both', 'do', 'then', 'is', 'just', 'but', 'so',
              'keep', 'what', 'yes', 'not', 'or', 'ill', 'we', 'are', 'with',
              'at', 'it', 'three', 'basketball', 'be', 'my', 'each', '-', "i'm",
              'could', 'your', 'in', 'on', 'too'}


def count_words_for_each_reward(file_name):
    """

        file_name: 'data.txt'

    Returns:
    """
    agree_reward_dict_list = [OrderedDict() for _ in range(11)]
    word_dict = OrderedDict()
    count_list = [0] * 11
    word_count_list = [0] * 11
    with open(file_name, 'r') as fd:
        for line in fd:
            tokens = line.strip().split()
            agree = tokens[-12] != 'no'
            if agree:
                reward = tokens[-8].split('=')[1]
                if reward.isdigit():
                    reward = int(reward)
                    count_list[reward] += 1
                    word_count_list[reward] += len(tokens[6:-12])
                else:
                    agree = False
            else:
                reward = 0
            tokens = tokens[6:-12]
            for token in tokens:
                if token in {'thanks', 'sounds', 'basketballs', 'books', 'balls', 'hats'}:
                    token = token.rstrip('s')
                word_dict[token] = word_dict.get(token, 0) + 1
                if agree:
                    agree_reward_dict_list[reward][token] = \
                        agree_reward_dict_list[reward].get(token, 0) + 1
    return word_dict, agree_reward_dict_list, count_list, word_count_list


def main():
    path = 'data/negotiate/'
    word_dict, agree_reward_dict_list, count_list, word_count_list = count_words_for_each_reward(path + 'data.txt')
    print(sorted(word_dict.items(), key=lambda x: x[1], reverse=True)[500])
    exit(0)
    total_num = sum(count_list)
    print('total agree num:', total_num)
    '''for word, count in sorted_list:
        p = count / total_num
        if p > 0.05:
            print('\t', word, ':', count, 'percentage', p)
        else:
            break'''
    print('agree:')
    for i in range(11):
        print('reward: ' + str(i))
        print('dialog num: ' + str(count_list[i]))
        tf_idf_dict = {}
        for word, count in agree_reward_dict_list[i].items():
            tf = count / word_count_list[i]
            idf = np.log(total_num / (word_dict[word] + 1))
            tf_idf_dict[word] = tf * idf
        tf_idf_sorted = sorted(tf_idf_dict.items(), key=lambda x: x[1], reverse=True)
        for j in range(min(50, len(tf_idf_sorted))):
            word, tf_idf = tf_idf_sorted[j]
            print('\t', word, 'TF-IDF: ', tf_idf)
        print('-' * 20)
    exit(0)
    thank_percent = []
    for i in range(4, 11):
        thank_percent.append(agree_reward_dict_list[i]['hi'] / count_list[i])
    x_label = ['0, 1, 2, 3, 4'] + [str(i) for i in range(5, 11)]
    import matplotlib.pyplot as plt
    plt.plot(thank_percent, '.')
    plt.title('hi')
    plt.xticks(range(len(thank_percent)), x_label)
    plt.xlabel('reward')
    plt.ylabel('percent')
    plt.show()


if __name__ == '__main__':
    main()
