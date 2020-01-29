#!/usr/bin/env python3.6

import os
import numpy as np


pred_folder="/media/rosu/Data/data/semantic_kitti/predictions/after_icra_experiments_fixed_deform_none/test"

out_folder="/media/rosu/Data/data/semantic_kitti/for_server/after_icra_experiments_fixed_deform_none"


#inside the pred folder we must go though all of the sequences and read the .label and then write it to binary
sequences = [ f.path for f in os.scandir(pred_folder) if f.is_dir() ]
print("sequences is ", sequences)
for seq_folder in sequences:
    seq=os.path.basename(seq_folder)
    out_folder_with_sequences=os.path.join(out_folder, "sequences", seq, "predictions" )
    print("out_folder_with_sequences ", out_folder_with_sequences)
    os.makedirs(out_folder_with_sequences, exist_ok=True)
    files = [f for f in os.listdir(seq_folder) if os.path.isfile(os.path.join(seq_folder, f))]
    nr_files_for_seq=0
    for file_basename in files:
        file=os.path.join(seq_folder, file_basename)
        extension = os.path.splitext(file)[1]
        if extension==".label":
            nr_files_for_seq+=1
            labels = np.loadtxt(file)
            out_file=os.path.join(out_folder_with_sequences, file_basename)
            # print("writing in", out_file)
            f= open(out_file,"w+")
            labels=labels.astype(np.int32)
            labels.tofile(f)

            #sanity check 
            # a = np.fromfile(out_file, dtype=np.uint32)
            # print("a is ", a)
            # print("labels is ", labels)
            # diff = (a!=labels).sum()
            # print("diff is", diff)
    print("nr_file_for_seq", nr_files_for_seq)


