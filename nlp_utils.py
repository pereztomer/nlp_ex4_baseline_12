from project_evaluate import read_file


def read_ds(path):
    ds = []
    file_en, file_de = read_file(path)
    for en_sen, gr_sen in zip(file_en, file_de):
        ds.append({'de': gr_sen, 'en': en_sen})

    return {'translation': ds}


def read_ds_unlabeled(path):
    ds = []
    with open(path) as my_file:
        for line in my_file:
            if line == 'German:\n':
                german_sample = []
            elif line == '\n':
                new_sample = {'Roots in English': roots,
                              'Modifiers in English': modifiers,
                              'gr': german_sample}
                ds.append(new_sample)
            elif 'Roots in English:' in line:
                roots = line
            elif 'Modifiers in English:' in line:
                modifiers = line
            else:
                german_sample.append(line)

    return ds
