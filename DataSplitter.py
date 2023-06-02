import numpy as np
import math
import scipy.sparse as sp


def read_data_file(train_file, valid_file, test_file, info_file, implicit, UIRT):
    """
    Reads data files.
    Returns {train, valid, test} matrices, dictionary of train data with number of users and items.

    :param str train_file: Filepath of train data
    :param str test_file: Filepath of test data
    :param bool implicit: Boolean indicating if rating should be converted to 1

    :return int num_users: Number of users
    :return int num_items: Number of items
    :return np.dok_matrix train_matrix: (num_users, num_items) shaped matrix with training ratings are stored
    :return np.dok_matrix test_matrix: (num_users, num_items) shaped matrix with test ratings are stored
    :return dict train_dict: Dictionary of training data. Key: Value = User id: List of related items.
    """

    # Read the meta file.
    separator = '\t'
    #with open(info_file+"_total", "r") as f:
        # The first line is the basic information for the dataset.
    #    num_users, num_items, num_ratings = list(map(int, f.readline().split(separator)))
    
    # ML1M
    # num_users = 6041
    # num_items = 3953
    # num_ratings = 1000209

    # ML10M
    # num_users = 73000
    # num_items = 10677
    # num_ratings = 10000054

    # Yelp
    num_users = 25677
    num_items = 25815
    num_ratings = 731671

    # Build training and test matrices.
    train_dict = [[] for _ in range(num_users)]
    train_matrix = sp.dok_matrix((num_users, num_items), dtype=np.float32)
    valid_matrix = sp.dok_matrix((num_users, num_items), dtype=np.float32)
    test_matrix = sp.dok_matrix((num_users, num_items), dtype=np.float32)

    UIRT = False

    # Read the training file.
    print("Loading the train data from \"%s\"" % train_file)
    with open(train_file, "r") as f:
        for line in f.readlines():
            if UIRT:
                u, i, r, t = line.strip().split(separator)
                user_id, item_id, rating, time = int(u)-1, int(i)-1, float(r), int(t)
            else:
                u, i, r = line.strip().split(separator)
                user_id, item_id, rating = int(u)-1, int(i)-1, float(r)
            if implicit:
                # if rating >= 4:    # Added for reconducting experiments in Table 2 (230521)
                #     rating = 1
                # else:
                #     rating = 0
                rating = 1
            #print(user_id, item_id, train_matrix.shape)
            train_dict[user_id].append([item_id, rating])
            train_matrix[user_id, item_id] = rating

    # Read the valid file.
    print("Loading the valid data from \"%s\"" % valid_file)
    with open(valid_file, "r") as f:
        for line in f.readlines():
            if UIRT:
                u, i, r, t = line.strip().split(separator)
                user_id, item_id, rating, time = int(u), int(i), float(r), int(t)
            else:
                u, i, r = line.strip().split(separator)
                user_id, item_id, rating = int(u)-1, int(i)-1, float(r)
            if implicit:
                if rating >= 4:    # Added for reconducting experiments in Table 2 (230521)
                    rating = 1
                else:
                    continue
                    # rating = 0

                # rating = 1
            else:
                if rating < 4:
                    rating = 0
            valid_matrix[user_id, item_id] = rating

    # Read the test file.
    print("Loading the test data from \"%s\"" % test_file)
    with open(test_file, "r") as f:
        for line in f.readlines():
            if UIRT:
                u, i, r, t = line.strip().split(separator)
                user_id, item_id, rating, time = int(u), int(i), float(r), int(t)
            else:
                u, i, r = line.strip().split(separator)
                user_id, item_id, rating = int(u)-1, int(i)-1, float(r)
            if implicit:
                if rating >= 4:    # Added for reconducting experiments in Table 2 (230521)
                    rating = 1
                else:
                    continue
                    # rating = 0

                # rating = 1
            else:
                if rating < 4:
                    rating = 0
            test_matrix[user_id, item_id] = rating

    print("\"num_users\": %d, \"num_items\": %d, \"num_ratings\": %d" % (num_users, num_items, num_ratings))
    return num_users, num_items, train_matrix, valid_matrix, test_matrix, train_dict


def save_leave_one_out(data_file, info_file, separator, popularity_order=True, UIRT=False):
    """
    Read data and split it into train, valid and test in leave-one-out manner.

    :param str data_file: File path of data to read
    :param str info_file: File path of data information to save
    :param str separator: String by which UIRT line is seperated
    :param bool popularity_order:

    :return: None
    """
    # Read the data and reorder it by popularity.
    num_users, num_items, num_ratings, user_ids, item_ids, UIRTs_per_user = order_by_popularity(data_file, separator, popularity_order, UIRT)

    num_ratings_per_user, num_ratings_per_item = {}, {}
    new_user_ids, new_item_ids = {}, {}

    # Assign new user_id for each user.
    for cnt, u in enumerate(user_ids):
        new_user_ids[u[0]] = cnt
        num_ratings_per_user[cnt] = u[1]
    # Assign new item_id for each item.
    for cnt, i in enumerate(item_ids):
        new_item_ids[i[0]] = cnt
        num_ratings_per_item[cnt] = i[1]
    # Convert UIRTs with new user_id and item_id.
    for u in UIRTs_per_user.keys():
        for UIRT in UIRTs_per_user[u]:
            i = UIRT[1]
            UIRT[0] = str(new_user_ids[u])
            UIRT[1] = str(new_item_ids[i])

    # Build info lines, user_idx_lines, and item_idx_lines.
    info_lines = []
    info_lines.append('\t'.join([str(num_users), str(num_items), str(num_ratings)]))
    info_lines.append("Sparsity : %.2f%%" % ((1 - (num_ratings / (num_users * num_items))) * 100))
    ratings_per_user = list(num_ratings_per_user.values())
    info_lines.append("Min/Max/Avg. ratings per users : %d %d %.2f" %
                      (min(ratings_per_user), max(ratings_per_user), np.mean(ratings_per_user)))
    ratings_per_item = list(num_ratings_per_item.values())
    info_lines.append("Min/Max/Avg. ratings per items : %d %d %.2f" %
                      (min(ratings_per_item), max(ratings_per_item), np.mean(ratings_per_item)))
    info_lines.append('User_id\tNumber of ratings')
    for u in range(num_users):
        info_lines.append("\t".join([str(u), str(num_ratings_per_user[u])]))
    info_lines.append('\nItem_id\tNumber of ratings')
    for i in range(num_items):
        info_lines.append("\t".join([str(i), str(num_ratings_per_item[i])]))

    with open(info_file + '_total', 'w') as f:
        f.write('\n'.join(info_lines))

    user_idx_lines, item_idx_lines = [], []
    user_idx_lines.append('Original_user_id\tCurrent_user_id')
    for u, v in user_ids:
        user_idx_lines.append("\t".join([str(u), str(new_user_ids[u])]))
    item_idx_lines.append('Original_item_id\tCurrent_item_id')
    for i, v in item_ids:
        item_idx_lines.append("\t".join([str(i), str(new_item_ids[i])]))

    with open(info_file + '_user_id', 'w') as f:
        f.write('\n'.join(user_idx_lines))
    with open(info_file + '_item_id', 'w') as f:
        f.write('\n'.join(item_idx_lines))
    print("Save leave-one-out files.")


def read_sequence_data_file(train_file, valid_file, test_file, info_file, sequence_len, target_len):
    """
    Reads data files.
    Returns {train, valid, test} matrices, dictionary of train data with number of users and items.

    :param str train_file: Filepath of train data
    :param str valid_file: Filepath of valid data
    :param str test_file: Filepath of test data
    :param str info_file: Filepath of data info
    :param int sequence_len: Length of sequence for model input
    :param int target_len: Length of target for model output

    :return int num_users: Number of users
    :return int num_items: Number of items
    :return int num_train_seqs: Number of train sequences
    :return int num_test_seqs: Number of test sequences
    :return np.dok_matrix train_users: (num_users) shaped matrix with training ratings are stored
    :return np.dok_matrix train_sequences: (num_users, sequence_len) shaped matrix with validation ratings are stored
    :return np.dok_matrix train_targets: (num_users, target_len) shaped matrix with test ratings are stored
    :return dict train_dict: Dictionary of training data. Key: Value = User id: List of related items.
    """

    # Read the meta file.
    separator = '\t'
    with open(info_file, "r") as f:
        # The first line is the basic information for the dataset.
        num_users, num_items, num_ratings, num_train_sequences, num_test_sequences = \
            list(map(int, f.readline().split(separator)))

    # Build training and test matrices.
    train_users = np.zeros(num_train_sequences, dtype=int)
    train_sequences = np.zeros((num_train_sequences, sequence_len), dtype=int)
    train_targets = np.zeros((num_train_sequences, target_len), dtype=int)
    valid_users = np.zeros(num_test_sequences, dtype=int)
    valid_sequences = np.zeros((num_test_sequences, sequence_len), dtype=int)
    valid_targets = np.zeros((num_test_sequences, target_len), dtype=int)
    test_users = np.zeros(num_test_sequences, dtype=int)
    test_sequences = np.zeros((num_test_sequences, sequence_len), dtype=int)
    test_targets = np.zeros((num_test_sequences, target_len), dtype=int)

    # Read the training file.
    print("Loading the train data from \"%s\"" % train_file)
    with open(train_file, "r") as f:
        for i, line in enumerate(f.readlines()):
            user, seq, tar = line.strip().split(separator)
            user_id = int(user)
            sequence_id = [int(x) for x in seq.split(',')]
            if ',' in tar:
                target_id = [int(x) for x in tar.split(',')]
            else:
                target_id = int(tar)

            train_users[i] = user_id
            train_sequences[i] = sequence_id
            train_targets[i] = target_id

    # Read the valid file.
    print("Loading the valid data from \"%s\"" % valid_file)
    with open(valid_file, "r") as f:
        for i, line in enumerate(f.readlines()):
            user, seq, tar = line.strip().split(separator)
            user_id = int(user)
            sequence_id = [int(x) for x in seq.split(',')]
            if ',' in tar:
                target_id = [int(x) for x in tar.split(',')]
            else:
                target_id = int(tar)

            valid_users[i] = user_id
            valid_sequences[i] = sequence_id
            valid_targets[i] = target_id

    # Read the test file.
    print("Loading the test data from \"%s\"" % test_file)
    with open(test_file, "r") as f:
        for i, line in enumerate(f.readlines()):
            user, seq, tar = line.strip().split(separator)
            user_id = int(user)
            sequence_id = [int(x) for x in seq.split(',')]
            if ',' in tar:
                target_id = [int(x) for x in tar.split(',')]
            else:
                target_id = int(tar)

            test_users[i] = user_id
            test_sequences[i] = sequence_id
            test_targets[i] = target_id

    print("\"num_users\": %d, \"num_items\": %d, \"num_ratings\": %d" % (num_users, num_items, num_ratings))
    return num_users, num_items, num_train_sequences, num_test_sequences, \
           train_users, train_sequences, train_targets, \
           valid_users, valid_sequences, valid_targets,\
           test_users, test_sequences, test_targets


def save_sequence_leave_one_out(data_file, train_file, valid_file, test_file, info_file, separator, sequence_len,
                                target_len,
                                popularity_order=True):
    """
    Read data and split it into train, valid and test in leave-one-out manner.

    :param str data_file: File path of data to read
    :param str train_file: File path of train data to save
    :param str valid_file: File path of valid data to save
    :param str test_file: File path of test data to save
    :param str info_file: File path of data information to save
    :param str separator: String by which UIRT line is seperated
    :param int sequence_len: Length of sequence for model input
    :param int target_len: Length of target for model output

    :param bool popularity_order:

    :return: None
    """
    # Read the data and reorder it by popularity.
    num_users, num_items, num_ratings, user_ids, item_ids, UIRTs_per_user \
        = order_by_popularity(data_file, separator, popularity_order)

    num_ratings_per_user, num_ratings_per_item = {}, {}
    new_user_ids, new_item_ids = {}, {}

    # Assign new user_id for each user.
    for cnt, u in enumerate(user_ids):
        new_user_ids[u[0]] = cnt
        num_ratings_per_user[cnt] = u[1]
    # Assign new item_id for each item.
    for cnt, i in enumerate(item_ids):
        new_item_ids[i[0]] = cnt
        num_ratings_per_item[cnt] = i[1]
    # Convert UIRTs with new user_id and item_id.
    for u in UIRTs_per_user.keys():
        for UIRT in UIRTs_per_user[u]:
            i = UIRT[1]
            UIRT[0] = str(new_user_ids[u])
            UIRT[1] = str(new_item_ids[i])

    # Build train and test lines.
    train_lines, valid_lines, test_lines = [], [], []
    num_valid_users = 0
    num_test_users = 0
    for u in UIRTs_per_user.keys():
        # Sort the UIRTs by the descending order of the timestamp.
        UIRTs_per_user[u] = sorted(UIRTs_per_user[u], key=lambda x: x[-1])
        # For valid, test dataset
        if len(UIRTs_per_user[u]) > (sequence_len + target_len + 1):
            num_test_users += 1
            last_sequences = UIRTs_per_user[u][-(sequence_len + target_len):-(target_len)]
            last_targets = UIRTs_per_user[u][-(target_len):]
            for _ in range(target_len):
                UIRTs_per_user[u].pop()
            # test user
            test_UST = [str(new_user_ids[u])]

            # test sequence
            test_sequences = []
            for sequence_UIRT in last_sequences:
                test_sequences.append(sequence_UIRT[1])
            test_UST.append(','.join(test_sequences))

            # test target
            test_targets = []
            for target_UIRT in last_targets:
                test_targets.append(target_UIRT[1])
            test_UST.append(','.join(test_targets))

            test_lines.append('\t'.join(test_UST))

            num_valid_users += 1
            last_sequences = UIRTs_per_user[u][-(sequence_len + target_len):-(target_len)]
            last_targets = UIRTs_per_user[u][-(target_len):]
            for _ in range(target_len):
                UIRTs_per_user[u].pop()
            # valid user
            valid_UST = [str(new_user_ids[u])]

            # valid sequence
            valid_sequences = []
            for sequence_UIRT in last_sequences:
                valid_sequences.append(sequence_UIRT[1])
            valid_UST.append(','.join(valid_sequences))

            # target
            valid_targets = []
            for target_UIRT in last_targets:
                valid_targets.append(target_UIRT[1])
            valid_UST.append(','.join(valid_targets))

            valid_lines.append('\t'.join(valid_UST))  # For train dataset
        # For train dataset
        for i in range(len(UIRTs_per_user[u]) - (target_len + sequence_len)):
            cur_seqs = UIRTs_per_user[u][i:i + sequence_len]
            cur_targets = UIRTs_per_user[u][i + sequence_len:i + sequence_len + target_len]

            # train user
            train_UST = [str(new_user_ids[u])]
            # train sequence
            train_sequences = []
            for sequence_UIRT in cur_seqs:
                train_sequences.append(sequence_UIRT[1])
            train_UST.append(','.join(train_sequences))
            # train target
            train_targets = []
            for target_UIRT in cur_targets:
                train_targets.append(target_UIRT[1])
            train_UST.append(','.join(train_targets))

            train_lines.append('\t'.join(train_UST))

    # Build info lines, user_idx_lines, and item_idx_lines.
    info_lines, user_idx_lines, item_idx_lines = [], [], []
    info_lines.append(
        '\t'.join([str(num_users), str(num_items), str(num_ratings), str(len(train_lines)), str(len(test_lines))]))
    info_lines.append("Sparsity : %.2f%%" % ((1 - (num_ratings / (num_users * num_items))) * 100))
    ratings_per_user = list(num_ratings_per_user.values())
    info_lines.append("Min/Max/Avg. ratings per users : %d %d %.2f" %
                      (min(ratings_per_user), max(ratings_per_user), np.mean(ratings_per_user)))
    ratings_per_item = list(num_ratings_per_item.values())
    info_lines.append("Min/Max/Avg. ratings per items : %d %d %.2f" %
                      (min(ratings_per_item), max(ratings_per_item), np.mean(ratings_per_item)))
    info_lines.append("Number of users in valid set : %d" % (num_valid_users))
    info_lines.append("Number of users in test set : %d" % (num_test_users))
    info_lines.append('User_id\tNumber of ratings')
    for u in range(num_users):
        info_lines.append("\t".join([str(u), str(num_ratings_per_user[u])]))
    info_lines.append('\nItem_id\tNumber of ratings')
    for i in range(num_items):
        info_lines.append("\t".join([str(i), str(num_ratings_per_item[i])]))

    user_idx_lines.append('Original_user_id\tCurrent_user_id')
    for u, v in user_ids:
        user_idx_lines.append("\t".join([str(u), str(new_user_ids[u])]))
    item_idx_lines.append('Original_item_id\tCurrent_item_id')
    for i, v in item_ids:
        item_idx_lines.append("\t".join([str(i), str(new_item_ids[i])]))

    # Save train, test, info, user_idx, item_idx files.
    with open(train_file, 'w') as f:
        f.write('\n'.join(train_lines))
    with open(valid_file, 'w') as f:
        f.write('\n'.join(valid_lines))
    with open(test_file, 'w') as f:
        f.write('\n'.join(test_lines))
    with open(info_file, 'w') as f:
        f.write('\n'.join(info_lines))
    with open(info_file + '_user_id', 'w') as f:
        f.write('\n'.join(user_idx_lines))
    with open(info_file + '_item_id', 'w') as f:
        f.write('\n'.join(item_idx_lines))
    print("Save Sequence leave-one-out files.")


def order_by_popularity(data_file, separator, popularity_order=True, UIRT=False):
    """
    Reads data file.
    Returns Item-Rating-Time, dictionary of train data with number of users and items.

    :param data_file:
    :param separator:
    :param popularity_order:

    :return int num_users:
    :return int num_items
    :return int num_ratings
    :return int sorted_user_ids, sorted_item_ids, UIRTs_per_user
    :return int sorted_item_ids, UIRTs_per_user
    :return int UIRTs_per_user
    """
    num_users, num_items, num_ratings = 0, 0, 0
    user_ids, item_ids, UIRTs_per_user = {}, {}, {}

    # Read the data file.
    print("Loading the dataset from \"%s\"" % data_file)
    with open(data_file, "r") as f:
        # Format (user_id, item_id, rating, timestamp)
        for line in f.readlines():
            #print(line)
            if UIRT:
                user_id, item_id, rating, time = line.strip().split(separator)
            else:
                user_id, item_id, rating = line.strip().split(separator)
            user_id, item_id = user_id, item_id

            # Update the number of ratings per user
            if user_id not in user_ids:
                user_ids[user_id] = 1
                UIRTs_per_user[user_id] = []
                num_users += 1
            else:
                user_ids[user_id] += 1

            # Update the number of ratings per item
            if item_id not in item_ids:
                item_ids[item_id] = 1
                num_items += 1
            else:
                item_ids[item_id] += 1

            num_ratings += 1
            #line = [str(user_id), str(item_id), str(rating), str(time)]
            line = [str(user_id), str(item_id), str(rating)]
            UIRTs_per_user[user_id].append(line)
    print("\"num_users\": %d, \"num_items\": %d, \"num_ratings\": %d" % (num_users, num_items, num_ratings))

    if popularity_order:
        # Sort the user_ids and item_ids by the popularity
        sorted_user_ids = sorted(user_ids.items(), key=lambda x: x[-1], reverse=True)
        sorted_item_ids = sorted(item_ids.items(), key=lambda x: x[-1], reverse=True)
    else:
        sorted_user_ids = user_ids.items()
        sorted_item_ids = item_ids.items()

    return num_users, num_items, num_ratings, sorted_user_ids, sorted_item_ids, UIRTs_per_user
