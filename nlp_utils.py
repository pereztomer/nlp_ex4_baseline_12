from project_evaluate import read_file


def read_ds(path):
    ds = []
    file_en, file_de = read_file(path)
    for en_sen, gr_sen in zip(file_en, file_de):
        ds.append({'de':gr_sen, 'en':en_sen})

    return {'translation':ds}
