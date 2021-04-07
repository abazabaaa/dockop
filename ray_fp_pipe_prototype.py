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
    
def batch_to_parquet(chunk):
    print(type(chunk))
    df = pa.RecordBatch.to_pandas(chunk)
    table = pa.Table.from_pandas(df)
    target_directory = '/data/dockop_data/feathers_zinc_15'
    name = os.path.join(target_directory,'zinc_15_subset%s'+ '.feather')
    unique_feather = next_path(name)
    with open(unique_feather, 'wb') as f:
        feather.write_feather(table, f)
    print(f'The feather was written to {unique_feather}')

def gen_fp(mols, pars, fingerprint_function):
    mols_b = copy.deepcopy(mols)
    mol_name = []
    fp_arr = []

    for mol in mols_b:
        row_idx = list()
        col_idx = list()
        count = 0
        fp = fingerprint_function(mol, **pars)
        onbits = list(fp.GetOnBits())
        col_idx+=onbits
        row_idx += [count]*len(onbits)
        unfolded_size = 8192
        fingerprint_matrix = sparse.coo_matrix((np.ones(len(row_idx)).astype(bool), (row_idx, col_idx)), 
                                       shape=(max(row_idx)+1, unfolded_size))
        fingerprint_matrix =  sparse.csr_matrix(fingerprint_matrix)
        x = sparse.csr_matrix.toarray(fingerprint_matrix)
        name = mol.GetProp('_Name')
        mol_name.append(name)
        fp_arr.append(x[0])
    data = fp_arr
    batch = pa.RecordBatch.from_arrays(data, mol_name)
    x = pa.RecordBatch.to_pandas(batch)
    tab = pa.Table.from_pandas(x)
    print(f'tab made successfully')
    return tab

@ray.remote
def count_ligs(chunk):
    print(type(chunk))
    print(chunk.schema)
    table_batch = chunk
    smiles = list(table_batch.column('smiles'))
    names = list(table_batch.column('zinc_id'))
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
    named_mols = []
    for count,m in enumerate(smiles):
            mol = Chem.MolFromSmiles(str(m))
            molid = names[count]
            mol.SetProp('_Name', str(molid))
            named_mols.append(mol)
    record_batches = gen_fp(named_mols, pars, fingerprint_function)
    tab = gen_fp(named_mols, pars, fingerprint_function)
    target_directory = '/data/dockop_data/feathers_zinc_15'
    name = os.path.join(target_directory,'zinc_15_subset%s'+ '.feather')
    unique_feather = next_path(name)
    with open(unique_feather, 'wb') as f:
        feather.write_feather(tab, f)
    print(f'The feather was written to {unique_feather}')
    return unique_feather

def csv_chunk_extractor(chunksize, include_columns):
    #select the CSV of interest.
    filename = '/home/schrogpu/code_test/dockop/data/DCAD.smi'
    #open_csv is single threaded, so usethreads doesn't do anthing.
    opts = pa.csv.ReadOptions(use_threads=True, block_size=chunksize)
    
    #Choose the correc delimiter
    parse_options= pa.csv.ParseOptions(delimiter=' ')

    #Only read in needed columns. This saves memory
    convert_options=pa.csv.ConvertOptions(include_columns=include_columns)
    table = pa.csv.open_csv(filename, opts, parse_options, convert_options)

    #We use futures to act as a list of results from each arrow batch. arr is
    #the numpy array with the reference fingerprint.
    futures = [count_ligs.remote(chunk) for chunk in table]

    #We use this to deserialize results from the rayworkers.
    results = [ray.get(f) for f in futures]
    print(f'here is the type of a slice of the results: {type(results)}')
    return results

ray.init(num_cpus=30)
pyarrow.set_cpu_count(30)
include_columns = ['smiles', 'zinc_id']
chunksize = 1048576
table_list = csv_chunk_extractor(chunksize, include_columns)

print(results)
