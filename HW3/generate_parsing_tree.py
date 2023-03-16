import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from chu_liu_edmonds import decode_mst
from project_evaluate import read_file
import json


# def padding_(sentences, seq_len):
#     features = np.zeros((len(sentences), seq_len), dtype=int)
#     for ii, review in enumerate(sentences):
#         if len(review) != 0:
#             features[ii, :len(review)] = np.array(review)[:seq_len]
#     return features


# class CustomDataset(Dataset):
#     def __init__(self, sentences, positions, seq_len_vals):
#         self.sentences = sentences
#         self.positions = positions
#         self.seq_len_values = seq_len_vals
#
#     def __len__(self):
#         return len(self.sentences)
#
#     def __getitem__(self, idx):
#         return self.sentences[idx], self.positions[idx], self.seq_len_values[idx]


# def predict(model, data_loader, device):
#     model.eval()
#     predictions = []
#     with torch.no_grad():
#         for x, pos, real_seq_len in data_loader:
#             x = torch.squeeze(x)[:real_seq_len].to(device)
#             pos = torch.squeeze(pos)[:real_seq_len].to(device)
#             real_seq_len = torch.squeeze(real_seq_len).to(device)
#             sample_score_matrix = model(padded_sentence=x,
#                                         padded_pos=pos,
#                                         real_seq_len=real_seq_len)
#
#             mst, _ = decode_mst(sample_score_matrix.detach().cpu().numpy(), sample_score_matrix.shape[0],
#                                 has_labels=False)
#             predictions.append(mst)
#     return predictions


def splits_paragraph(paragraph):
    sen_splits_temp = paragraph.split('.')
    sen_splits = []
    for sp in sen_splits_temp:
        if len(sp) == 1 or sp == '':
            continue
        elif sp[0] == ' ' and sp[-1] == ' ':
            sen_splits.append(sp[1:-1])
        elif sp[-1] == ' ':
            sen_splits.append(sp[:-1])
        elif sp[0] == ' ':
            sen_splits.append(sp[1:])
        else:
            sen_splits.append(sp)

    split_ds = []
    for sp in sen_splits:
        # if sp == '' or len(sp) == 1:
        #     continue
        # else:
        sub_splits = sp.split('?')
        for sub_sp in sub_splits:
            if sub_sp == '' or len(sub_sp) == 1:
                continue
            elif sub_sp[0] == ' ' and sub_sp[-1] == ' ':
                split_ds.append(sub_sp[1:-1])
            elif sub_sp[-1] == ' ':
                split_ds.append(sub_sp[:-1])
            elif sub_sp[0] == ' ':
                split_ds.append(sub_sp[1:])
            else:
                split_ds.append(sub_sp)
    return split_ds


def main():
    model = torch.load('comp_model_mlp_ex3').to('cuda')
    sentences_word2idx = model.sentences_word2idx

    file_en, file_de = read_file('/home/user/PycharmProjects/nlp_ex4_baseline_12/data/val.labeled')
    generated_samples = []

    for counter, (en_pr, ger_pr) in enumerate(zip(file_en, file_de)):
        if counter % 1000 == 0:
            print(f'{counter}/{len(file_en)}')

        eg_paragraph_temp = en_pr.replace('\n', '')
        en_sentences = splits_paragraph(eg_paragraph_temp)
        split_en_paragraph_list = []
        for sen in en_sentences:
            words = sen.split(' ')
            words.insert(0, 'ROOT')

            embedded_words = [sentences_word2idx[word] if word in sentences_word2idx else 1 for word in words]
            sen_positions = [0] * len(words)
            sentence_real_len = len(words)

            model.eval()
            with torch.no_grad():
                x = torch.Tensor(embedded_words)
                x = x.int()
                x = x.to('cuda')
                pos = torch.Tensor(sen_positions)
                pos = pos.int()
                pos = pos.to('cuda')
                sample_score_matrix = model(padded_sentence=x,
                                            padded_pos=pos,
                                            real_seq_len=sentence_real_len)

                mst, _ = decode_mst(sample_score_matrix.detach().cpu().numpy(), sample_score_matrix.shape[0],
                                    has_labels=False)

            split_en_paragraph_list.append((words, mst.tolist()))

        paragraphs_dict = {'en': en_pr, 'de': ger_pr, 'parsing_tree': split_en_paragraph_list}
        generated_samples.append(paragraphs_dict)

    with open(f"/home/user/PycharmProjects/nlp_ex4_baseline_12/data"
              f"/val_dependency_parsed.json", "w") as outfile:
        outfile.write(json.dumps(generated_samples, indent=4))


if __name__ == '__main__':
    main()
