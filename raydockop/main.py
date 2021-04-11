import sys
from set_up import Setup
from estimator import CommonEstimator
import json
import h5py
import glob
import os
from scipy import sparse
import numpy as np
#from utils import get_memory_usage
import pyarrow as pa
import numpy as np
import pyarrow.feather as feather
import pandas as pd
SEED = 12939 #from random.org
np.random.seed(SEED)

# def write_results(preds, fpsize, trainingSize, name, repeat_number):
#         """Writes an HDF5 file that stores the results. 
#         preds: np.array: prediction scores for the test samples
#         fpsize: int: size the fingerprint was folded to
#         name: str: the estimator name, as stored in the json
#         repeat_number: int.
 
#         Results stored are:
#         - test indices
#         - preds 
#         and there should be one set of results for each repeat."""

#         #write the first time, append afterwards. 
#         write_option = 'w' if repeat_number==0 else 'a'
#         outf = h5py.File('../processed_data/'+self.fingerprint_kind+'_'+str(fpsize)+'_'+str(trainingSize)+'_'+name+'.hdf5', write_option)

#         rp = outf.create_group(f'repeat{repeat_number}')

#         dset_idx = rp.create_dataset('test_idx', self.test_idx.shape, dtype='int')
#         dset_idx[:] = self.test_idx

#         dset_pred = rp.create_dataset('prediction', preds.shape, dtype='float16')
#         dset_pred[:] = preds
        
#         outf.close()

def fold_fingerprints(feature_matrix):
    """Folds a fingerprint matrix by bitwise OR.
    (scipy will perform the bitwise OR because the `data` is bool,
    and it will not cast it to int when two Trues are added."""

    ncols = feature_matrix.shape[1]
    return feature_matrix[:,:ncols//2] + feature_matrix[:,ncols//2:]

def fold_to_size(size, fingerprints):
    """Performs the `fold` operation multiple times to reduce fp 
    length to the desired size."""
    feature_matrix = fingerprints
    while feature_matrix.shape[1]>size:
        feature_matrix = fold_fingerprints(feature_matrix)
    return feature_matrix

def random_split(self, number_train_ligs):
    """Simply selects some test and train indices"""



fpType = sys.argv[1]
fpSize = int(sys.argv[2])
trainingSetSize = int(sys.argv[3])
json_name = sys.argv[4]
dataset = sys.argv[5]


print('Running:')
print(f'python main.py {fpType} {fpSize} {json_name} {dataset}')


estimators = json.load(open(json_name, 'r'))['estimators']

if __name__=='__main__':
    #setup the data:
    setup = Setup(fpType, dataset, verbose=True)
    # try:
    #     setup.write_fingerprints()
    # except:
    #     print('Already written fpfile')        
    # setup.load_fingerprints()
    # setup.load_scores()
    
    dataset = '/data/newdockop/dockop/code/mod_code_base/data_out/testprocessed_data'
# +input_db_ext)
    


    fingerprint_file_ext = ".npz"
    scores_file_ext = ".feather"
    fingerprint_file_names_list = glob.glob(os.path.join(dataset+"*"+fingerprint_file_ext))

    fingerprint_files_list = [(dataset+'{:01d}'.format(x)+ fingerprint_file_ext) for x in range(len(fingerprint_file_names_list))]
    scores_files_list = [(dataset+'{:01d}'.format(y)+ scores_file_ext) for y in range(len(fingerprint_file_names_list))]

    npz_list = []
    scores_list = []
    names_list = []
    smiles_list = []
    for batch_num in range(300):
        fingerprints = sparse.load_npz(fingerprint_files_list[batch_num])
        df = feather.read_feather(scores_files_list[batch_num])
        scores = list(df['scores'])
        smiles = list(df['smiles'])
        names = list(df['names'])
        npz_list.append(fingerprints)
        scores_list.append(scores)
        names_list.append(names)
        smiles_list.append(smiles)

    flat_sparse_fingerprints = sparse.vstack(npz_list)

    flat_scores_list = [item for sublist in scores_list for item in sublist]
    flat_names_list = [item for sublist in names_list for item in sublist]
    flat_smiles_list = [item for sublist in smiles_list for item in sublist]
    scores_arry = np.array(scores_list, dtype=np.float16)
    np_scores = np.concatenate(scores_arry)
    num_ligs = len(flat_scores_list)

    feature_matrix = fold_to_size(fpSize, flat_sparse_fingerprints)
    
    #evaluation stuff goes here:    
    for estimator in estimators:

        for repeat in range(5):
            idx = np.arange(num_ligs)

            np.random.shuffle(idx)


            train_idx = idx[:trainingSetSize]
            test_idx = idx[trainingSetSize:]

            # training_smi = [flat_smiles_list[i] for i in train_idx]
            test_smi = [flat_smiles_list[i] for i in test_idx]


            # training_names = [flat_names_list[i] for i in train_idx]
            test_names = [flat_names_list[i] for i in test_idx]

            # training_scores = [flat_scores_list[i] for i in train_idx]
            test_scores = [flat_scores_list[i] for i in test_idx]
            
            common_estimator = CommonEstimator(estimator, cutoff=0.3, verbose=True)
            print(train_idx.shape)
            print(np_scores.shape)
            common_estimator.fit(feature_matrix[train_idx], np_scores[train_idx])
            pred = common_estimator.chunked_predict(feature_matrix[test_idx])
            pred_list = [pred[i] for i in range(len(pred))]
            print(f'length of prediction list is {len(pred_list)}')
            print(f'length of smiles is {len(test_smi)}')
            print(f'length of names is {len(test_names)}')
            print(f'length of scores is {len(test_scores)}')

            # scores = [scores_list[i] for i in range(len(scores_list))]

            pred_list_pa = pa.array(pred_list)   
            smiles_pa = pa.array(test_smi, type=pa.string())
            scores_pa = pa.array(test_scores)
            names_pa = pa.array(test_names, type=pa.string())

            data = [
            pred_list_pa,
            smiles_pa,
            scores_pa,
            names_pa
            ]

            batch_from_data = pa.RecordBatch.from_arrays(data, ['pred_list', 'smiles', 'scores', 'names'])
            df = batch_from_data.to_pandas()
            feather.write_feather(df, f'test{repeat}.feather')

            # setup.write_results(pred, fpSize, trainingSetSize, estimator['name'], repeat, test_idx)


        # idx_pre_shuffled_pa = pa.array(idx_list_pre_shuffle, type=pa.int64())        
        # idx_shuffled_pa = pa.array(idx_list_shuffled, type=pa.int64())

        # data = [
        # pred_list_pa,
        # idx_pre_shuffled_pa,
        # idx_shuffled_pa,
        # smiles_pa,
        # scores_pa,
        # names_pa
        # ]
