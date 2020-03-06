import pickle


if __name__ == '__main__':

    file_name = 'data/cdr_data/data_tree.pkl'
    with open(file_name,'rb') as f:
        data_tree = pickle.load(f)

    count = 0
    for key,trees in data_tree.items():
        if count != 1:
            count +=1
            continue
        else:
            for tree in trees:
                print('.....................')
                for dep,tok1,tok2 in tree:
                    print(dep,tok1,tok2)

            break