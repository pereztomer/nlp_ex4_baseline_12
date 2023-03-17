import torch
from transformers import pipeline
from nlp_utils import load_ds_unlabeled_modifiers


def main():
    model_name = 'w_modifiers_roots_t5-base'
    new_file_path = f'models/{model_name}/val.labeled_{model_name}'
    unlabeled_ds = load_ds_unlabeled_modifiers(path='data/val.unlabeled')
    for val in unlabeled_ds:
        intro_sen = ''
        new_dict = {}
        for root, modifiers in zip(val['Roots in English'], val['Modifiers in English']):
            modifiers_to_add = ''
            for m in modifiers:
                modifiers_to_add = modifiers_to_add + m + ', '
            intro_sen += f'sentence root: {root}, root modifiers: {modifiers_to_add}'

        zero_entry = intro_sen + ' German sentences to translate: '
        val['gr'].insert(0, zero_entry)

    translator = pipeline("translation", model=f'models/{model_name}/checkpoint-77500', device='cuda:0')
    sen_to_translate_lst = []
    for idx, val in enumerate(unlabeled_ds):
        sen_to_translate = "translate German to English: "
        for sen in val['gr']:
            sen_to_translate += sen
        sen_to_translate_lst.append(sen_to_translate)

    translations = translator(sen_to_translate_lst, max_length=420)

    with open(new_file_path, "w") as new_file:
        for idx, (val, translated_eng_sen) in enumerate(zip(unlabeled_ds, translations)):
            new_file.write('German:\n')
            for counter, val_2 in enumerate(val['gr']):
                if counter == 0:
                    continue
                else:
                    new_file.write(val_2)
            new_file.write('English:\n')
            english_split_sen = translated_eng_sen['translation_text'].split('.')
            for eng_sen in english_split_sen:
                if eng_sen != '':
                    new_file.write(eng_sen + '.\n')
            new_file.write('\n')


if __name__ == '__main__':
    main()
