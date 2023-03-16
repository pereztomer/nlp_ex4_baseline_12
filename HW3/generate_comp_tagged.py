import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from chu_liu_edmonds import decode_mst


def padding_(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len), dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, :len(review)] = np.array(review)[:seq_len]
    return features


class CustomDataset(Dataset):
    def __init__(self, sentences, positions, seq_len_vals):
        self.sentences = sentences
        self.positions = positions
        self.seq_len_values = seq_len_vals

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.positions[idx], self.seq_len_values[idx]


def predict(model, data_loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for x, pos, real_seq_len in data_loader:
            x = torch.squeeze(x)[:real_seq_len].to(device)
            pos = torch.squeeze(pos)[:real_seq_len].to(device)
            real_seq_len = torch.squeeze(real_seq_len).to(device)
            sample_score_matrix = model(padded_sentence=x,
                                        padded_pos=pos,
                                        real_seq_len=real_seq_len)

            mst, _ = decode_mst(sample_score_matrix.detach().cpu().numpy(), sample_score_matrix.shape[0],
                                has_labels=False)
            predictions.append(mst)
    return predictions


def main():
    # comp_address = '/home/user/PycharmProjects/nlp_ex_3/val_untaged.txt'
    comp_address = './data/comp.unlabeled'
    model = torch.load('comp_model_mlp_ex3').to('cuda')
    print('here')
    sentences_word2idx = model.sentences_word2idx
    pos_word2idx = model.pos_word2idx
    comp_sentences, comp_sentence_positions, comp_sentences_real_len = parse_comp_file(comp_address)
    max_sen_len = 0
    comp_sentences_idx = []
    for sent in comp_sentences:
        if len(sent) > max_sen_len:
            max_sen_len = len(sent)
        comp_sentences_idx.append(
            [sentences_word2idx[word] if word in sentences_word2idx else 1 for word in sent])

    comp_pos_idx = []
    for sent_pos in comp_sentence_positions:
        comp_pos_idx.append([pos_word2idx[pos] if pos in pos_word2idx else 0 for pos in sent_pos])

    comp_sentences_idx_padded = padding_(comp_sentences_idx, 250)
    comp_comp_pos_idx = padding_(comp_pos_idx, 250)

    comp_ds = CustomDataset(sentences=comp_sentences_idx_padded,
                            positions=comp_comp_pos_idx,
                            seq_len_vals=comp_sentences_real_len)

    comp_data_loader = DataLoader(dataset=comp_ds,
                                  batch_size=1,
                                  shuffle=False)
    predictions = predict(model, comp_data_loader, 'cuda')
    write_file(comp_address, predictions)


if __name__ == '__main__':
    main()
