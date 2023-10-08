from ast import literal_eval 
import scipy
from scipy import stats
import os

dataset = "mredplus"
root_dir = "results/model_analysis/"



with open(os.path.join(root_dir, dataset+"_bl.txt"), "r", encoding="utf-8") as f:
    lines = f.readlines()
    bl_enc_std = literal_eval(lines[1].strip())
    bl_enc_doc = literal_eval(lines[3].strip())
    bl_dec_std = literal_eval(lines[5].strip())

with open(os.path.join(root_dir, dataset+"_hierencdec.txt"), "r", encoding="utf-8") as f:
    lines = f.readlines()
    ours_enc_std = literal_eval(lines[1].strip())
    ours_enc_doc = literal_eval(lines[3].strip())
    ours_dec_std = literal_eval(lines[5].strip())

if len(bl_enc_std) == len(bl_enc_std) == 200:
    enc_std_stats = scipy.stats.ttest_ind(bl_enc_std, ours_enc_std, equal_var=False)
    print("encoder analysis:")
    print("bl avg:", sum(bl_enc_std)/len(bl_enc_std))
    print("ours avg:", sum(ours_enc_std)/len(ours_enc_std))
    print(enc_std_stats)
    print('\n')

if len(bl_enc_std) == len(bl_enc_std) == 200:
    enc_doc_stats = scipy.stats.ttest_ind(bl_enc_doc, ours_enc_doc, equal_var=False)
    print("encoder analysis:")
    print("bl avg:", sum(bl_enc_doc)/len(bl_enc_doc))
    print("ours avg:", sum(ours_enc_doc)/len(ours_enc_doc))
    print(enc_doc_stats)
    print('\n')

if len(bl_dec_std) == len(ours_dec_std) == 200:
    dec_std_stats = scipy.stats.ttest_ind(bl_dec_std, ours_dec_std, equal_var=False)
    print("dec_stdoder analysis:\n")
    print("bl avg:", sum(bl_dec_std)/len(bl_dec_std))
    print("ours avg:", sum(ours_dec_std)/len(ours_dec_std))
    print(dec_std_stats)
    print('\n')