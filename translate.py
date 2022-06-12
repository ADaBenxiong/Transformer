''' Translate input text with trained model. '''

import torch
import argparse
import dill as pickle
from tqdm import tqdm

import transformer.Constants as Constants
from torchtext.data import Dataset
from transformer.Models import Transformer
from transformer.Translator import Translator


def load_model(opt, device):

    checkpoint = torch.load(opt.model, map_location=device)
    model_opt = checkpoint['settings']

    model = Transformer(
        model_opt.src_vocab_size,
        model_opt.trg_vocab_size,

        model_opt.src_pad_idx,
        model_opt.trg_pad_idx,

        trg_emb_prj_weight_sharing=model_opt.proj_share_weight,
        emb_src_trg_weight_sharing=model_opt.embs_share_weight,
        d_k=model_opt.d_k,
        d_v=model_opt.d_v,
        d_model=model_opt.d_model,
        d_word_vec=model_opt.d_word_vec,
        d_inner=model_opt.d_inner_hid,
        n_layers=model_opt.n_layers,
        n_head=model_opt.n_head,
        dropout=model_opt.dropout).to(device)

    model.load_state_dict(checkpoint['model'])
    print('[Info] Trained model state loaded.')
    return model 


def main():
    '''Main Function'''

    parser = argparse.ArgumentParser(description='translate.py')

    parser.add_argument('-model', required=True,
                        help='Path to model weight file')
    parser.add_argument('-data_pkl', required=True,
                        help='Pickle file with both instances and vocabulary.')
    parser.add_argument('-output', default='pred.txt',
                        help="""Path to output the predictions (each line will
                        be the decoded sequence""")
    parser.add_argument('-beam_size', type=int, default=5)
    parser.add_argument('-max_seq_len', type=int, default=100)
    parser.add_argument('-no_cuda', action='store_true')

    # TODO: Translate bpe encoded files 
    #parser.add_argument('-src', required=True,
    #                    help='Source sequence to decode (one line per sequence)')
    #parser.add_argument('-vocab', required=True,
    #                    help='Source sequence to decode (one line per sequence)')
    # TODO: Batch translation
    #parser.add_argument('-batch_size', type=int, default=30,
    #                    help='Batch size')
    #parser.add_argument('-n_best', type=int, default=1,
    #                    help="""If verbose is set, will output the n_best
    #                    decoded sentences""")

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    print(opt)

    data = pickle.load(open(opt.data_pkl, 'rb'))
    SRC, TRG = data['vocab']['src'], data['vocab']['trg']

    print(len(SRC.vocab.stoi))
    print(len(TRG.vocab.stoi))

    opt.src_pad_idx = SRC.vocab.stoi[Constants.PAD_WORD]#填充符

    opt.trg_pad_idx = TRG.vocab.stoi[Constants.PAD_WORD]#填充符
    opt.trg_bos_idx = TRG.vocab.stoi[Constants.BOS_WORD]#起始符
    opt.trg_eos_idx = TRG.vocab.stoi[Constants.EOS_WORD]#终止符

    # print(TRG.vocab.itos[1])
    # print(TRG.vocab.itos[2])
    # print(TRG.vocab.itos[3])


    print(opt)

    test_loader = Dataset(examples=data['test'], fields={'src': SRC, 'trg': TRG})

    device = torch.device('cuda' if opt.cuda else 'cpu')

    translator = Translator(
        model=load_model(opt, device),
        beam_size=opt.beam_size,
        max_seq_len=opt.max_seq_len,
        src_pad_idx=opt.src_pad_idx,
        trg_pad_idx=opt.trg_pad_idx,
        trg_bos_idx=opt.trg_bos_idx,
        trg_eos_idx=opt.trg_eos_idx).to(device)

    unk_idx = SRC.vocab.stoi[SRC.unk_token]

#=============================================================add====================================================
    # print("*" * 100)
    # print(unk_idx)
    #
    # print(test_loader)
    # for example in test_loader:
    #     print(example)
    #     for j in example.src:
    #         print(j, end = " ")
    #     print("")
    #
    #     for j in example.trg:
    #         print(j, end = " ")
    #     print("")
    #     # print(example.src)
    #     # print(example.trg)
    #     break
    #
    # print("*" * 100)
    #
    # for example in test_loader:
    #     src_seq = [SRC.vocab.stoi.get(word, unk_idx) for word in example.src]
    #     print(src_seq)
    #     pred_seq = torch.LongTensor([src_seq]).to(device)   #德语转换为数字
    #     pred_seq_ans = translator.translate_sentence(torch.LongTensor([src_seq]).to(device))    #德语经过模型翻译成英语数字
    #     print("prediction:")
    #     #print(pred_seq_ans)
    #     pred_line = ' '.join(TRG.vocab.itos[idx] for idx in pred_seq_ans)
    #     pred_line = pred_line.replace(Constants.BOS_WORD, '').replace(Constants.EOS_WORD, '')
    #     print(pred_line)
    #     compare1 = torch.tensor(pred_seq_ans[1: -1]).contiguous()
    #     print(type(compare1))
    #     print(compare1)
    #
    #     print("answer:")
    #     ans = [word for word in example.trg]
    #     answer = ' '.join(ans)
    #     print(answer)
    #     print("-"* 100)
    #     compare2 = torch.tensor([TRG.vocab.stoi.get(word, unk_idx) for word in example.trg]).contiguous()
    #     print(type(compare2))
    #     print(compare2)
    #
    #
    #     # print(compare1)
    #     # print(compare2)
    #
    #     length = max(len(compare1), len(compare2))
    #     if len(compare1) < length:
    #         add = torch.zeros(length - len(compare1)).long()
    #         compare1 = torch.cat((compare1, add), 0)
    #
    #     if len(compare2) < length:
    #         add = torch.zeros(length - len(compare2)).long()
    #         compare2 = torch.cat((compare2, add), 0)
    #
    #     print(compare1)
    #     print(compare2)
    #     n_correct = compare1.eq(compare2).sum().item()  # masked_select 选出是True的值
    #     print(n_correct)
    #     break
# =============================================================add====================================================
    n_word_total, n_word_correct = 0, 0


    with open(opt.output, 'w') as f:
        for example in tqdm(test_loader, mininterval=2, desc='  - (Test)', leave=False):
            #print(' '.join(example.src))

            src_seq = [SRC.vocab.stoi.get(word, unk_idx) for word in example.src] #将字符转换成为数字
            pred_seq = translator.translate_sentence(torch.LongTensor([src_seq]).to(device)) #预测的答案数字
            pred_line = ' '.join(TRG.vocab.itos[idx] for idx in pred_seq)       #预测的答案转换为字符
            # print(pred_line)    #输出
            pred_line = pred_line.replace(Constants.BOS_WORD, '').replace(Constants.EOS_WORD, '') #预测的答案转换为字符去掉首尾
            # print(pred_line)    #输出
            #print(pred_line)
            # print(example.src)  #输入
            # print(src_seq)      #输入
            # print(pred_seq)     #输出
            # print(pred_line)    #输出
            # print(us)
            f.write(pred_line.strip() + '\n')

            compare_prediction = torch.tensor(pred_seq[1: -1]).contiguous()
            compare_answer = torch.tensor([TRG.vocab.stoi.get(word, unk_idx) for word in example.trg]).contiguous()
            length = max(len(compare_prediction), len(compare_answer))
            #length = max(len(compare_prediction), len(compare_answer))
            if len(compare_prediction) < length:
                add = torch.zeros(length - len(compare_prediction)).long()
                compare_prediction = torch.cat((compare_prediction, add), 0)

            if len(compare_answer) < length:
                add = torch.zeros(length - len(compare_answer)).long()
                compare_answer = torch.cat((compare_answer, add), 0)

            n_correct = compare_prediction.eq(compare_answer).sum().item()
            n_word = length
            n_word_total += n_word
            n_word_correct += n_correct

    print(n_word_correct / n_word_total)
    with open(opt.output, 'a') as f:
        f.write("accuracy is : " + str(n_word_correct/ n_word_total) + '\n')

    print('[Info] Finished.')


if __name__ == "__main__":
    '''
    Usage: python translate.py -model trained.chkpt -data multi30k.pt -no_cuda
    '''
    main()
