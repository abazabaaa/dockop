
from scipy import sparse
import h5py

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from tqdm import tqdm
from pathlib import Path
import numpy as np

class Setup(object):
    """Handles all the evaluation stuff for a given fingerprint setting."""
    def __init__(self, fingerprint, smifile, verbose=False):
        """This class just wraps all the analysis together so that it's easier later to 
        evaluate multiple fingerprint types and regressors/classifiers using a common interface. 


        Parameters
        -----------
        fingerprint: str. One of: 'morgan'
        fpsize: int. Dimensions of the fingerprint. Rdkit will do the folding down to this size.
        smifile: str. A text file with a single column and a header. Each line below the header 
        is a single smiles code for a ligand. This comes from parse_data.py"""
        
        self.fingerprint_kind=fingerprint
        #these two come from parse_data.py
        self.base = smifile
        self.smifile = self.base+'_short.smi'
        self.scorefile = self.base+'_short.npy'
        # self.num_ligs = sum(1 for line in open(self.smifile))-1 #it comes in handy a few times to know how many ligands there are
        self.verbose=verbose

    # def load_smiles(self):
    #     """Loads the smifile and stores as list """
    #     if self.verbose:
    #         print('loading smiles')

    #     f = open(self.smifile, 'r')
    #     f.readline()
    #     self.smiles = np.array([line[:-1] for line in f])
    #     f.close()

    # def load_scores(self):
    #     """Loads the scores and stores as np.float16"""
    #     self.scores = np.load(self.scorefile)
    
    # def get_fingerprint_function(self):
    #     """RDKit has lots of different ways to make fingerprits. 
    #     So this just returns the correct function for a given FP.

    #     Source of parameters is (awesome) FPSim2 from ChEMBL: 
    #     https://github.com/chembl/FPSim2/blob/master/FPSim2/io/chem.py

    #     No input since the fingerprint type is set during init"""

    #     if self.fingerprint_kind=='morgan':
    #         function = rdMolDescriptors.GetMorganFingerprintAsBitVect
    #         pars = { "radius": 2,
    #                  "nBits": 65536,
    #                  "invariants": [],
    #                  "fromAtoms": [],
    #                  "useChirality": False,
    #                  "useBondTypes": True,
    #                  "useFeatures": False,
    #         }
    #     if self.fingerprint_kind=='morgan_feat':
    #         function = rdMolDescriptors.GetMorganFingerprintAsBitVect
    #         pars = { "radius": 2,
    #                  "nBits": 65536,
    #                  "invariants": [],
    #                  "fromAtoms": [],
    #                  "useChirality": False,
    #                  "useBondTypes": True,
    #                  "useFeatures": True,
    #         }
    #     if self.fingerprint_kind=='atompair':
    #         function = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect
    #         pars = { "nBits": 65536,
    #                  "minLength": 1,
    #                  "maxLength": 30,
    #                  "fromAtoms": 0,
    #                  "ignoreAtoms": 0,
    #                  "atomInvariants": 0,
    #                  "nBitsPerEntry": 4,
    #                  "includeChirality": False,
    #                  "use2D": True,
    #                  "confId": -1,
    #         }
    #     if self.fingerprint_kind=='topologicaltorsion':
    #         function = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect
    #         pars = { "nBits": 65536,
    #                  "targetSize": 4,
    #                  "fromAtoms": 0,
    #                  "ignoreAtoms": 0,
    #                  "atomInvariants": 0,
    #                  "includeChirality": False,
    #         }
    #     if self.fingerprint_kind=='maccs':
    #         function = rdMolDescriptors.GetMACCSKeysFingerprint
    #         pars = { }
    #     if self.fingerprint_kind=='rdk':
    #         function = Chem.RDKFingerprint
    #         pars = { "minPath": 1,
    #                 "maxPath": 6, #reduced this from 7 to reduce numOnBits
    #                 "fpSize": 65536,
    #                 "nBitsPerHash": 1, #reduced from 2 to reduce numOnBits
    #                 "useHs": True,
    #                 "tgtDensity": 0.0,
    #                 "minSize": 128,
    #                 "branchedPaths": True,
    #                 "useBondOrder": True,
    #                 "atomInvariants": 0,
    #                 "fromAtoms": 0,
    #                 "atomBits": None,
    #                 "bitInfo": None,
    #         }
    #     if self.fingerprint_kind=='pattern':
    #         function = Chem.PatternFingerprint
    #         pars = { "fpSize": 65536,
    #                  "atomCounts": [],
    #                  "setOnlyBits": None
    #         }
            
    #     return function, pars


    # def write_fingerprints(self, overWrite=False):
    #     """Writes one of the rdkit fingerprints to a sparse matrix.
    #     Currently using size 65536 - this is usually way too large, 
    #     but it leaves room to move. There is a folding function to
    #     get back to common usage sizes. 

    #     This function also checks if a fingerprint file has been written
    #     already. If so, if requires `overWrite` to be True to re-write
    #     the file. 
    #     """

    #     fingerprint_file = Path("../processed_data/"+self.base+'_'+self.fingerprint_kind+".npz")
    #     if fingerprint_file.is_file() and not overWrite:
    #         raise Exception('Fingerprint file exists already. Set `overWrite` to true to re-write it')
    #     else:
    #         pass
        
    #     if self.verbose:
    #         print('Generating fingerprints at size 65536 (except MACCS)...')

    #     fingerprint_function, pars = self.get_fingerprint_function()

    #     smifile = open(self.smifile, 'r') #file containing the smiles codes.
    #     smifile.readline() #read past the header.

    #     #store bit indices in these:
    #     row_idx = list()
    #     col_idx = list()
        
    #     #iterate through file, 
    #     for count, line in tqdm(enumerate(smifile), total=self.num_ligs, smoothing=0):
    #         mol = Chem.MolFromSmiles(line[:-1])
    #         fp = fingerprint_function(mol, **pars)
    #         onbits = list(fp.GetOnBits())
    #         #these bits all have the same row:
    #         row_idx += [count]*len(onbits)
    #         #and the column indices of those bits:
    #         col_idx+=onbits
            
    #     smifile.close()
        
    #     #generate a sparse matrix out of the row,col indices:
    #     unfolded_size = 65536
    #     fingerprint_matrix = sparse.coo_matrix((np.ones(len(row_idx)).astype(bool), (row_idx, col_idx)), 
    #                       shape=(max(row_idx)+1, unfolded_size))
    #     #convert to csr matrix, it is better:
    #     fingerprint_matrix =  sparse.csr_matrix(fingerprint_matrix)

    #     #save file:
    #     sparse.save_npz('../processed_data/'+self.base+'_'+self.fingerprint_kind+'.npz', fingerprint_matrix)


    # def load_fingerprints(self):
    #     """Load the npz file saved in the `write_fingerprints` step. 
    #     """

    #     fingerprint_file = Path("../processed_data/"+self.base+'_'+self.fingerprint_kind+".npz")
    #     if not fingerprint_file.is_file():
    #         raise Exception('Fingerprint file does not exists already. Run `write_fingerprints`')

    #     if self.verbose:
    #         print('loading fingerprints npz file')

    #     #use sparse fingerprints:
    #     self.fingerprints = sparse.load_npz('../processed_data/'+self.base+'_'+self.fingerprint_kind+'.npz')

        
    # def fold_fingerprints(self, feature_matrix):
    #     """Folds a fingerprint matrix by bitwise OR.
    #     (scipy will perform the bitwise OR because the `data` is bool,
    #     and it will not cast it to int when two Trues are added."""

    #     ncols = feature_matrix.shape[1]
    #     return feature_matrix[:,:ncols//2] + feature_matrix[:,ncols//2:]

    # def fold_to_size(self, size):
    #     """Performs the `fold` operation multiple times to reduce fp 
    #     length to the desired size."""

    #     if self.verbose:
    #         print(f'Folding to {size}...')
    #     if self.fingerprint_kind=='MACCS':
    #         return self.fingerprints
        
    #     feature_matrix = self.fingerprints
    #     while feature_matrix.shape[1]>size:
    #         feature_matrix = self.fold_fingerprints(feature_matrix)

    #     return feature_matrix
        
        
    # def random_split(self, number_train_ligs):
    #     """Simply selects some test and train indices"""
    #     idx = np.arange(self.num_ligs)
    #     np.random.shuffle(idx)
    #     self.train_idx = idx[:number_train_ligs]
    #     self.test_idx = idx[number_train_ligs:]

        
    def write_results(self, preds, fpsize, trainingSize, name, repeat_number, test_idx):
        """Writes an HDF5 file that stores the results. 
        preds: np.array: prediction scores for the test samples
        fpsize: int: size the fingerprint was folded to
        name: str: the estimator name, as stored in the json
        repeat_number: int.
 
        Results stored are:
        - test indices
        - preds 
        and there should be one set of results for each repeat."""

        #write the first time, append afterwards. 
        write_option = 'w' if repeat_number==0 else 'a'
        outf = h5py.File('/data/newdockop/dockop/code/mod_code_base/data_out/'+self.fingerprint_kind+'_'+str(fpsize)+'_'+str(trainingSize)+'_'+name+'.hdf5', write_option)

        rp = outf.create_group(f'repeat{repeat_number}')

        dset_idx = rp.create_dataset('test_idx', test_idx.shape, dtype='int')
        dset_idx[:] = test_idx

        dset_pred = rp.create_dataset('prediction', preds.shape, dtype='float16')
        dset_pred[:] = preds
        
        outf.close()
