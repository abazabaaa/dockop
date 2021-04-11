import os
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

# random.seed(733101) #from www.random.org

filename = sys.argv[1]
# desired_num_ligands = sys.argv[2]
outfilename = sys.argv[2]


#find out how many ligands there are that have an associated docking score:
print('Counting ligands with docking score...')
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
def namesdict_to_arrow_batch(name_list, scores_list, smiles_list):
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
@stopwatch
def fp_to_batch(record_batch, job_count):
#     print(type(table_batch))
#     print(table_batch)

    smiles = list(record_batch.column('smiles'))
    scores = list(record_batch.column('dockscore'))
    zincid = list(record_batch.column('zincid'))
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
        try:
            mol = Chem.MolFromSmiles(str(m))
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
            smiles_list.append(m)

        except:
            print('molecule failed')

    unfolded_size = 8192        
    fingerprint_matrix = sparse.coo_matrix((np.ones(len(row_idx)).astype(bool), (row_idx, col_idx)), 
              shape=(max(row_idx)+1, unfolded_size))
    fingerprint_matrix =  sparse.csr_matrix(fingerprint_matrix)

    target_directory = '/data/dockop_data/processed_data_ampc'
    name = os.path.join(outfilename+'processed_data'+str(job_count))
    # print(scores_list)
    # # scores_list = np.array(scores_list, dtype=np.float16)
    # print(scores_list)
    record_batch = namesdict_to_arrow_batch(name_list, scores_list, smiles_list)
    df = record_batch.to_pandas()
    # print(df['scores'])

    feather.write_feather(df, f'{name}.feather')
    sparse.save_npz(name+'.npz' , fingerprint_matrix)

    print(f'Job number {job_count} complete.')
    print(f'Job contained {len(smiles_list)} smiles strings')
    print(f'Job generated spares matrix with {len(row_idx)} row_idx')
    print(f'Job generated spares matrix with {len(col_idx)} col_idx')
    return job_count

# @ray.remote
# @stopwatch
# def count_ligs(chunk, job_count):
#     print(job_count)
#     print(type(chunk))
#     print(chunk.schema)
#     table_batch = chunk
#     smiles = list(table_batch.column('smiles'))
#     scores = list(table_batch.column('dockscore'))

#     pars = { "radius": 2,
#                  "nBits": 8192,
#                  "invariants": [],
#                  "fromAtoms": [],
#                  "useChirality": False,
#                  "useBondTypes": True,
#                  "useFeatures": False,
#                  }

#     fingerprint_function = rdMolDescriptors.GetMorganFingerprintAsBitVect        
#     print(f'the number of smiles in the record batch is {len(smiles)}')
#     count_ligs = len(smiles)
#     smiles = [x for x in smiles]
#     mols = []
#     scores_list = []
#     for count,m in enumerate(smiles):
#         try:
#             mol = Chem.MolFromSmiles(str(m))
#             score = str(scores[count])
#             scores_list.append(score)
#             mols.append(mol)
#         except:
#             print('molecule failed')

#     name_path = gen_fp(mols, pars, fingerprint_function, scores_list, job_count)


#     return name_path

ray.init(num_cpus=30)
pyarrow.set_cpu_count(20)
# source = '/data/dockop_data/AmpC_screen_table_clean.feather'
reader = pa.ipc.open_file(filename)


futures = [fp_to_batch.remote(reader.get_batch(i), i) for i in range(reader.num_record_batches)]
results = [ray.get(f) for f in futures]
# smifile = open(filename, 'r')
# smifile.readline() #read past the header.

# count = 0
# for line in smifile:
#     words = line[:-1].split(',')
#     if len(words[2])<1:
#         continue
#     if words[2]=='no_score':
#         break
#     count+=1
# smifile.close()

# print(f'There were {count} ligands')
# print('Randomly selecting and writing {desired_num_ligands}...')

# p = int(desired_num_ligands)/count


# smifile = open(filename, 'r')
# smifile.readline() #read past the header.


# outsmi.write('smiles\n')

# scores = list()

# for line in tqdm(smifile, total=count, smoothing=0):
#     if random.choices([True,False], weights=[p, 1-p])[0]:
#         words = line[17:-1].split(',') #removes the zinc ID and trailing newline
#         if len(words[1])<1:
#             continue
#         if words[1]=='no_score':
#             break
#         else:
#             scores.append(float(words[1]))
#             outsmi.write(words[0]+'\n')

# np.save(outfilename+'_newercode.npy', np.array(scores, dtype=np.float16))
# outsmi.close()
# smifile.close()
