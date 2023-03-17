from project_evaluate import read_file
import json


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


def load_parsed_ds(file_path):
    ds = json.load(open(file_path))
    out_ds = []
    for val in ds:
        new_dict = {'en': val['en']}

        intro_sen = ''
        for value in val['parsing_tree']:
            root_index = value[1].index(0)
            root = value[0][root_index]
            # get modifiers:
            modifiers_to_add = ''
            modifiers_of_root_indexes = [idx for idx, value_inner in enumerate(value[1]) if value_inner == root_index]
            if len(modifiers_of_root_indexes) >= 2:
                modifiers_to_add = modifiers_to_add + value[0][modifiers_of_root_indexes[0]]
                modifiers_to_add = modifiers_to_add + ', ' + value[0][modifiers_of_root_indexes[1]]
            elif len(modifiers_of_root_indexes) == 1:
                modifiers_to_add = modifiers_to_add + value[0][modifiers_of_root_indexes[0]]
            intro_sen += f'sentence root: {root}, root modifiers: {modifiers_to_add}, '

        new_dict['de'] = intro_sen + ' German sentences to translate: ' + val['de']
        out_ds.append(new_dict)
    return {'translation': out_ds}


def load_ds_unlabeled_modifiers(path):
    ds = []
    with open(path) as my_file:
        for line in my_file:
            if line == 'German:\n':
                german_sample = []
            elif line == '\n':
                new_sample = {'Roots in English': roots,
                              'Modifiers in English': modifiers_per_sentence_list,
                              'gr': german_sample}
                ds.append(new_sample)
            elif 'Roots in English:' in line:
                roots = line.replace('Roots in English:', '').replace('\n', '').replace(' ', '').split(',')
                if roots[-1] == '':
                    roots = roots[:-1]
            elif 'Modifiers in English:' in line:
                modifiers_per_sentence_list = []
                modifiers = line.replace('Modifiers in English: ', '').replace('\n', '').replace(' ', '').split('),')
                if modifiers[-1] == ' ':
                    modifiers = modifiers[:-1]
                for mod_sen in modifiers:
                    modifiers_in_sentence = mod_sen.replace('(', '').replace(')', '').replace(' ', '').split(',')
                    if modifiers_in_sentence[-1] == ' ':
                        modifiers_in_sentence = modifiers_in_sentence[:-1]
                    modifiers_per_sentence_list.append(modifiers_in_sentence)
            else:
                german_sample.append(line)

    return ds
