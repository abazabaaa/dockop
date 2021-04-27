mport os
import copy
import sys
import pyarrow as pa
from pyarrow import csv
import pandas as pd
import pyarrow.feather as feather
from rdkit.Chem import rdMolDescriptors
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np
from datetime import timedelta
from timeit import time
from scipy import sparse
import pyarrow.parquet as pq
import ray
import pyarrow.csv
import pyarrow.dataset as ds
import pathlib
# random.seed(733101) #from www.random.org

# filename = sys.argv[1]
# desired_num_ligands = sys.argv[2]
outfilename = sys.argv[1]


#find out how many ligands there are that have an associated docking score:
# print('Counting ligands with docking score...')
# read_df = feather.read_feather(filename)
# df = read_df['smiles']
# feather.write_feather(df, outfilename+'_newercode.smi')

def stopwatch(method):
    def timed(*args, **kw):
        ts = time.perf_counter()
        result = method(*args, **kw)
        te = time.perf_counter()
        duration = timedelta(seconds=te - ts)
        print(f"{method.__name__}: {duration}")
        return result
    return timed

# @stopwatch
# def gen_fp(mols, pars, fingerprint_function, scores_list, job_count):
#     mols_b = copy.deepcopy(mols)

#     row_idx = list()
#     col_idx = list()


#     for count,mol in enumerate(mols_b):
#         fp = fingerprint_function(mol, **pars)
#         onbits = list(fp.GetOnBits())
#         col_idx+=onbits
#         row_idx += [count]*len(onbits)

#     print(len(row_idx))
#     print(len(col_idx))
#     unfolded_size = 8192
#     fingerprint_matrix = sparse.coo_matrix((np.ones(len(row_idx)).astype(bool), (row_idx, col_idx)), 
#                       shape=(max(row_idx)+1, unfolded_size))
#     fingerprint_matrix =  sparse.csr_matrix(fingerprint_matrix)
#     target_directory = '/data/dockop_data/processed_data_ampc'
#     print(str(job_count))
#     name = os.path.join(outfilename+'processed_data'+str(job_count))
#     print(name)
#     sparse.save_npz(name+'.npz' , fingerprint_matrix)
#     np.save(name+'.npy', np.array(scores_list, dtype=np.float16))
#     print('files_saved')
#     return name

@stopwatch       
def namesdict_to_arrow_batch(name_list, smiles_list, scores_list):
    scores = [scores_list[i] for i in range(len(scores_list))]
    smiles_list2 = [str(smiles_list[i]) for i in range(len(smiles_list))]
    # scores_list = np.array(scores_list, dtype=np.float16)
    name_list = pa.array(name_list, type=pa.string())
    smiles_list3 = pa.array(smiles_list2)
    scores_list2 = pa.array(scores)
    data1 = [
        name_list,
        scores_list2,
        smiles_list3
    ]

    batch_from_names_scores = pa.RecordBatch.from_arrays(data1, ['names', 'scores', 'smiles'])
    return batch_from_names_scores

@ray.remote
def fp_to_batch(job_count, record_batch):

    # for x, record_batch in enumerate(record_batch_list):
    smiles = list(record_batch.column('standard_smiles'))
    scores = list(record_batch.column('docking_score'))
    zincid = list(record_batch.column('canonical_id'))
    pars = { "radius": 2,
             "nBits": 8192,
             "invariants": [],
             "fromAtoms": [],
             "useChirality": False,
             "useBondTypes": True,
             "useFeatures": False,
             }

    fingerprint_function = rdMolDescriptors.GetMorganFingerprintAsBitVect        
    print(f'the number of smiles in the record batch is {len(smiles)}')
    count_ligs = len(smiles)



    scores_list = []
    name_list =[]
    smiles_list = []
    row_idx = list()
    col_idx = list()
    num_on_bits = []
    for count,m in enumerate(smiles):
        m_in = str(m)
        mol = Chem.MolFromSmiles(m_in)
        fp = fingerprint_function(mol, **pars)
        score = str(scores[count])
        zincid_name = str(zincid[count])
        onbits = list(fp.GetOnBits())
        col_idx+=onbits
        row_idx += [count]*len(onbits)
        num_bits = len(onbits)
        num_on_bits.append(num_bits)
        scores_list.append(score)
        name_list.append(zincid_name)
        smiles_list.append(m_in)

        # except:
        #     print('molecule failed')

    unfolded_size = 8192        
    fingerprint_matrix = sparse.coo_matrix((np.ones(len(row_idx)).astype(bool), (row_idx, col_idx)), 
              shape=(max(row_idx)+1, unfolded_size))
    fingerprint_matrix =  sparse.csr_matrix(fingerprint_matrix)

    # target_directory = '/data/newdockop/dockop/code/mod_code_base/dopamine3'
    name = os.path.join(outfilename,'processed_data'+str(job_count))
    # print(scores_list)
    # # scores_list = np.array(scores_list, dtype=np.float16)
    # print(scores_list)
    record_batch = namesdict_to_arrow_batch(name_list, smiles_list, scores_list)
    df = record_batch.to_pandas()
    # print(df['scores'])

    feather.write_feather(df, f'{name}.feather')
    sparse.save_npz(name+'.npz' , fingerprint_matrix)

    print(f'Job number {job_count} complete.')
    print(f'Job contained {len(smiles_list)} smiles strings')
    print(f'Job generated spares matrix with {len(row_idx)} row_idx')
    print(f'Job generated spares matrix with {len(col_idx)} col_idx')
