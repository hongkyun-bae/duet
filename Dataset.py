import os
import unittest
from data_loader import DataSplitter
import numpy as np

class Dataset(object):
    def __init__(self, data_dir, dataset, separator, implicit, split_type="fcv", sequence_len=3, target_len=1, popularity_order=True):
        self.data_dir = data_dir
        self.data_name = dataset
        self.filename = os.path.join(self.data_dir, self.data_name, self.data_name + '.rating')
        self.split_type = split_type
        self.sequence_len = sequence_len
        self.target_len = target_len
        self.popularity_order = popularity_order


        train_file = "./" + self.data_dir + "/" + self.data_name + "/u1.base"
        valid_file = "./" + self.data_dir + "/" + self.data_name + "/u1.valid"
        test_file = "./" + self.data_dir + "/" + self.data_name + "/u1.test"
        info_file = "./" + self.data_dir + "/" + self.data_name + "/info"
        UIRT = True    # False for ml1m


        DataSplitter.save_leave_one_out(self.filename, info_file, separator, self.popularity_order, UIRT)

        self.num_users = 0
        self.num_items = 0
        self.train_matrix = None
        self.valid_matrix = None
        self.test_matrix = None
        self.train_dict = None
        self.neg_dict = None
        self.neg_user_dict = None

        if split_type == "fcv":
            self.num_users, self.num_items, self.train_matrix, self.valid_matrix, self.test_matrix, self.train_dict = DataSplitter.read_data_file(train_file, valid_file, test_file, info_file, implicit, UIRT)
        else:
            raise Exception("Please choose a splitter.")

    @property
    def neg_items(self):
        if self.neg_dict == None:
            self.neg_dict = {u: [] for u in range(self.num_users)}
            all_items = set(list(range(self.num_items)))
            for u, items in enumerate(self.train_dict):
                self.neg_dict[u] += list(all_items - set([x[0] for x in items]))

        return self.neg_dict

    @property
    def neg_users(self):
        if self.neg_user_dict == None:
            self.neg_user_dict = {u: [] for u in range(self.num_items)}
            all_users = set(list(range(self.num_users)))
            for i in range(self.num_items):
                pos_users = self.train_matrix[:, i].nonzero()[0].tolist()
                self.neg_user_dict[i] += list(all_users - set(pos_users))

        return self.neg_user_dict

    def switch_mode(self, MODE):
        if self.split_type == 'fcv':
            if MODE.lower() == 'valid':
                self.eval_input = self.train_matrix.toarray()
                self.eval_target = self.valid_matrix.toarray()
                self.mode = 'valid'
            elif MODE.lower() == 'test':
                self.eval_input = (self.train_matrix + self.valid_matrix).toarray()
                self.eval_target = self.test_matrix.toarray()
                self.mode = 'test'
        elif self.split_type == 'seq_loo':
            if MODE.lower() == 'valid':
                self.eval_input = np.zeros((self.num_test_seqs, self.num_items))
                self.eval_target = np.zeros((self.num_test_seqs, self.num_items))
                for i, [sequence, target] in enumerate(zip(self.valid_sequences, self.valid_targets)):
                    self.eval_input[i, sequence] = 1
                    self.eval_target[i, target] = 1
                self.mode = 'valid'
            elif MODE.lower() == 'test':
                self.eval_input = np.zeros((self.num_test_seqs, self.num_items))
                self.eval_target = np.zeros((self.num_test_seqs, self.num_items))
                for i, [sequence, target] in enumerate(zip(self.test_sequences, self.test_targets)):
                    self.eval_input[i, sequence] = 1
                    self.eval_target[i, target] = 1
                self.mode = 'test'
        else:
            raise ValueError('Choose correct dataset mode. (valid or test)')

    def __str__(self):
        ret = '\n============= [Dataset] =============\n'
        ret += 'Filename: %s\n' % self.filename
        ret += 'Split type: %s\n' % self.split_type
        if self.split_type == 'ratio':
            ret += 'Split ratio: %s\n' % str(self.split_ratio)
        ret += 'Popularity order: %s\n' % str(self.popularity_order)
        ret += '# of User, Items: %d, %d\n' % (self.num_users, self.num_items)
        return ret


class TestDataset(unittest.TestCase):
    def runTest(self):
        filename = "..\\data\\ml1m\\ml1m.rating"
        separator = "::"
        implicit = True
        split_type = "fcv"
        split_ratio = [0.8, 0.1, 0.1]
        popularity_order = False
        dataset = Dataset(filename, separator, implicit, split_type, split_ratio, popularity_order)


if __name__ == '__main__':
    unittest.main()
