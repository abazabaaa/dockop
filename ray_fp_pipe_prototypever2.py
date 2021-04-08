import os
import copy
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

def next_path(path_pattern):
    i = 1
    while os.path.exists(path_pattern % i):
        i = i * 2
    a, b = (i // 2, i)
    while a + 1 < b:
        c = (a + b) // 2 # interval midpoint
        a, b = (c, b) if os.path.exists(path_pattern % c) else (a, c)
    return path_pattern % b
    

def gen_fp(mols, pars, fingerprint_function, scores_list, job_count):
    mols_b = copy.deepcopy(mols)
    row_idx = list()
    col_idx = list()
    count=0

    for mol in mols_b:

        fp = fingerprint_function(mol, **pars)
        onbits = list(fp.GetOnBits())
        col_idx+=onbits
        row_idx += [count]*len(onbits)
        count+=1
    print(len(row_idx))
    print(len(col_idx))
    print(count)
    unfolded_size = 8192
    fingerprint_matrix = sparse.coo_matrix((np.ones(len(row_idx)).astype(bool), (row_idx, col_idx)), 
                      shape=(max(row_idx)+1, unfolded_size))
    fingerprint_matrix =  sparse.csr_matrix(fingerprint_matrix)
    target_directory = '/data/dockop_data/processed_data'
    print(str(job_count))
    name = os.path.join(target_directory,'processed_data'+str(job_count))
    print(name)
    sparse.save_npz(name+'.npz' , fingerprint_matrix)
    np.save(name+'.npy', np.array(scores_list) )
    print('files_saved')
    return name

@ray.remote
def count_ligs(chunk, job_count):
    print(job_count)
    print(type(chunk))
    print(chunk.schema)
    table_batch = chunk
    smiles = list(table_batch.column('smiles'))
    scores = list(table_batch.column('dockscore'))
    fingerprint_function = rdMolDescriptors.GetMorganFingerprintAsBitVect
    pars = { "radius": 2,
                     "nBits": 8192,
                     "invariants": [],
                     "fromAtoms": [],
                     "useChirality": False,
                     "useBondTypes": True,
                     "useFeatures": True,
            }

    fingerprint_function = rdMolDescriptors.GetMorganFingerprintAsBitVect        
    print(f'the number of smiles in the record batch is {len(smiles)}')
    count_ligs = len(smiles)
    smiles = [x for x in smiles]
    mols = []
    scores_list = []
    for count,m in enumerate(smiles):
        try:
            mol = Chem.MolFromSmiles(str(m))
            score = scores[count]
            scores_list.append(score)
            mols.append(mol)
        except:
            print('molecule failed')

    name_path = gen_fp(mols, pars, fingerprint_function, scores_list, job_count)


    return name_path

def csv_chunk_extractor(chunksize, include_columns):
    #select the CSV of interest.
    filename = '/data/dockop_data/AmpC_screen_table.csv'
    #open_csv is single threaded, so usethreads doesn't do anthing.
    opts = pa.csv.ReadOptions(use_threads=True, block_size=chunksize)
    
    #Choose the correc delimiter
    parse_options= pa.csv.ParseOptions(delimiter=',')

    #Only read in needed columns. This saves memory
    convert_options=pa.csv.ConvertOptions(include_columns=include_columns)
    table = pa.csv.open_csv(filename, opts, parse_options, convert_options)

    #We use futures to act as a list of results from each arrow batch. arr is
    #the numpy array with the reference fingerprint.
    futures = [count_ligs.remote(chunk,count) for count,chunk in enumerate(table)]

    #We use this to deserialize results from the rayworkers.
    results = [ray.get(f) for f in futures]
    print(f'here is the type of a slice of the results: {results}')
    return results

ray.init(num_cpus=3)
pyarrow.set_cpu_count(3)
include_columns = ['smiles', 'dockscore']
chunksize = 1048576*10
table_list = csv_chunk_extractor(chunksize, include_columns)

print(results)
