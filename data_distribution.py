from collections import OrderedDict
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

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
                    if reward <= 5:
                        reward = 5
                    else:
                        reward = 6
                    count_list[reward] += 1
                    word_count_list[reward] += len(tokens[6:-12])
                else:
                    agree = False
            else:
                reward = 0
            tokens = tokens[6:-12]
            for token in set(tokens):
                if agree:
                    word_dict[token] = word_dict.get(token, 0) + 1
                    agree_reward_dict_list[reward][token] = \
                        agree_reward_dict_list[reward].get(token, 0) + 1
    return word_dict, agree_reward_dict_list, count_list, word_count_list


def input_distribution():
    with open('data/negotiate/data.txt') as fd:
        lines = fd.readlines()
    count_dict = {}
    i = 0
    fd1 = open('data/same_background1.txt', 'w')

    while i + 1 < len(lines):
        input_numbers = tuple(lines[i].split()[:6])
        sign1 = lines[i].split()[6]
        sign2 = lines[i + 1].split()[6]
        input_numbers_next = tuple(lines[i + 1].split()[:6])
        if (input_numbers[0] == input_numbers_next[0]
            and input_numbers[2] == input_numbers_next[2]
            and input_numbers[4] == input_numbers_next[4]
            and {sign1, sign2} == {'YOU:', 'THEM:'}):
            input_number_pairs = tuple(lines[i].split()[:6] + lines[i + 1].split()[:6])
            if input_number_pairs in count_dict:
                count_dict[input_number_pairs] += 1
            else:
                count_dict[input_number_pairs] = 1
            if input_number_pairs == ('2', '1', '2', '1', '3', '2', '2', '0', '2', '5', '3', '0'):
                fd1.write(lines[i])
                fd1.write(lines[i + 1])
            i += 1
        else:
            i += 1
    sorted_list = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)
    print(sorted_list)
    count = 0
    count1 = 0
    count2 = 0
    pair_dict = {}
    k = 0
    for i, j in sorted_list:
        if j > 1:
            count += j
            count2 += 1
            if i not in pair_dict:
                pair_dict[i] = k
                k += 1
        count1 += j
    print(count, count1, count2, len(lines), k)
    pair_dict_text = {}
    i = 0
    while i + 1 < len(lines):
        input_numbers = tuple(lines[i].split()[:6])
        input_numbers_next = tuple(lines[i + 1].split()[:6])
        if (input_numbers[0] == input_numbers_next[0]
            and input_numbers[2] == input_numbers_next[2]
            and input_numbers[4] == input_numbers_next[4]):
            input_number_pairs = tuple(lines[i].split()[:6] + lines[i + 1].split()[:6])
            if input_number_pairs in pair_dict:
                if input_number_pairs in pair_dict_text:
                    pair_dict_text[input_number_pairs].append(lines[i])

    return
    count_dict = {}
    with open('data/same_background.txt', 'w') as fd1:

        for line in lines:
            input_numbers = tuple(line.split()[:6])
            if input_numbers == ('2', '1', '2', '1', '3', '2'):
                fd1.write(line + '\n')
            if input_numbers in count_dict:
                count_dict[input_numbers] += 1
            else:
                count_dict[input_numbers] = 0
    sorted_list = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)
    print(sorted_list)


def main():
    input_distribution()
    exit(0)
    path = 'data/negotiate/'
    word_dict, agree_reward_dict_list, count_list, word_count_list = count_words_for_each_reward(path + 'data.txt')
    total_num = sum(count_list)
    print('total agree num:', total_num)
    '''for word, count in sorted_list:
        p = count / total_num
        if p > 0.05:
            print('\t', word, ':', count, 'percentage', p)
        else:
            break'''
    print('agree:')
    for i in range(5, 7):
        print('reward: ' + str(i))
        print('dialog num: ' + str(count_list[i]))
        tf_idf_dict = {}
        mi_dict = {}
        #for word in ['deal', '<selection>', 'button', 'have', 'take', 'need', ',']:
            #count = agree_reward_dict_list[i][word]
        for word, count in agree_reward_dict_list[i].items():
            tf = count / word_count_list[i]
            idf = np.log(total_num / (word_dict[word] + 1))
            tf_idf_dict[word] = tf * idf
            n = total_num
            n_11 = count
            n_01 = count_list[i] - count
            n_10 = word_dict[word] - count
            n_00 = n - count_list[i] - n_10
            n_1_ = word_dict[word]
            n__1 = count_list[i]
            n_0_ = total_num - word_dict[word]
            n__0 = total_num - count_list[i]
            #print(word, n, n_11, n_01, n_10, n_00, n_1_, n__1, n_0_, n__0)
            mi_1 = (n_11 / n) * np.log2(n * n_11 / (n_1_ * n__1 + 1))
            mi_2 = (n_01 / n) * np.log2(n * n_01 / (n_0_ * n__1 + 1))
            mi_3 = (n_10 / n) * np.log2(n * n_10 / (n_1_ * n__0 + 1))
            mi_4 = (n_00 / n) * np.log2(n * n_00 / (n_0_ * n__0 + 1))
            mi = mi_1 + mi_2 + mi_3 + mi_4
            if np.isnan(mi):
                mi = 0
            mi_dict[word] = mi
        mi_sorted = sorted(mi_dict.items(), key=lambda x: x[1], reverse=True)
        for j in range(min(50, len(mi_sorted))):
            word, mi = mi_sorted[j]
            print('\t', word, 'MI: ', mi)
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
