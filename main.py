import numpy as np
import scipy.sparse as sp
import os
import pandas as pd
import click
import shutil

# want triu with csr

def load_ld_npz(ld_prefix):
    # load the SNPs metadata
    gz_file = '%s.gz' % (ld_prefix)
    df_ld_snps = pd.read_table(gz_file, sep='\s+')
    df_ld_snps.rename(columns={'rsid': 'SNP', 'chromosome': 'CHR', 'position': 'BP', 'allele1': 'A1', 'allele2': 'A2'},
                      inplace=True, errors='ignore')
    assert 'SNP' in df_ld_snps.columns
    assert 'CHR' in df_ld_snps.columns
    assert 'BP' in df_ld_snps.columns
    assert 'A1' in df_ld_snps.columns
    assert 'A2' in df_ld_snps.columns
    df_ld_snps.index = df_ld_snps['CHR'].astype(str) + '.' + df_ld_snps['BP'].astype(str) + '.' + df_ld_snps[
        'A1'] + '.' + df_ld_snps['A2']

    # load the LD matrix
    npz_file = '%s.npz' % (ld_prefix)
    try:
        R = sp.load_npz(npz_file).toarray()
        R += R.T
    except ValueError:
        raise IOError('Corrupt file: %s' % (npz_file))

    # create df_R and return it
    df_R = pd.DataFrame(R, index=df_ld_snps.index, columns=df_ld_snps.index)
    return df_R, df_ld_snps

def reduce_submatrix(sparse_mat, start_ind, end_ind, precision):
    submat = sparse_mat[start_ind:end_ind, start_ind:end_ind]

    return adjust_to_zero(submat, precision)

def adjust_to_zero(sparse_matrix, precision):
    if precision:
        nonzeros = sparse_matrix.nonzero()
        nonzero_mask = np.array(np.abs(sparse_matrix[nonzeros]) < precision)[0]
        rows = nonzeros[0][nonzero_mask]
        cols = nonzeros[1][nonzero_mask]
        sparse_matrix[rows, cols] = 0

    sparse_matrix.eliminate_zeros()
    return sparse_matrix




def convert(infile, outfile, block_size, precision):
    start_snip, end_snip = [int(x) for x in infile.split('_')[-2:]]
    dir = outfile + f'_{block_size}_{precision}/'
    if not os.path.exists(dir):
        os.mkdir(dir)
    sparse_mat = sp.triu(sp.load_npz(infile + '.npz').T, format='csr')
    # sparse_mat.setdiag(1) #leave as 0.5 for easier symmetric construction
    mat_size = sparse_mat.shape[0]

    shutil.copy(infile + '.gz', dir + 'metadata.gz')
    pd.DataFrame([start_snip, end_snip, block_size]).to_csv(dir + 'META', index=False, header=False)

    for i in range(0, mat_size, block_size):
        if i + block_size < mat_size:
            off_diagonal = sparse_mat[i:i + block_size, i + block_size:]

            off_diagonal = adjust_to_zero(off_diagonal, precision) #or is tocoo() better?
            print(f'Writing off diagonal {i}')
            sp.save_npz(dir + f'row_{i}', off_diagonal, compressed=True)
            print('Finished writing')

        reduced = reduce_submatrix(sparse_mat, i, i + block_size, precision) # allow precision?
        print(f'Writing block {i}')
        sp.save_npz(dir + f'block_{i}', reduced, compressed=True)
        print('Finished writing')

def get_metadata(dir):
    metadata = pd.read_csv(dir + '/META', header=None)
    start_snip = metadata[0][0]
    end_snip = metadata[0][1]
    block_size = metadata[0][2]
    return start_snip, end_snip, block_size

def load_snips_df(dir):
    #load the SNPs metadata
    gz_file = dir + '/metadata.gz'
    df_ld_snps = pd.read_table(gz_file, sep='\s+')
    df_ld_snps.rename(columns={'rsid':'SNP', 'chromosome':'CHR', 'position':'BP', 'allele1':'A1', 'allele2':'A2'}, inplace=True, errors='ignore')
    assert 'SNP' in df_ld_snps.columns
    assert 'CHR' in df_ld_snps.columns
    assert 'BP' in df_ld_snps.columns
    assert 'A1' in df_ld_snps.columns
    assert 'A2' in df_ld_snps.columns
    df_ld_snps.rename_axis('relative_pos', inplace=True)
    df_ld_snps.reset_index(inplace=True)
    df_ld_snps.index = df_ld_snps['CHR'].astype(str) + '.' + df_ld_snps['BP'].astype(str) + '.' + df_ld_snps['A1'] + '.' + df_ld_snps['A2']
    return df_ld_snps

# def get_value(dir, i, j):
#     if i == j: #should handle better
#         return 1
#     if i < j:
#         i,j = j,i
#
#     start_snip, block_size = get_metadata(dir)
#
#     assert start_snip <= i
#     assert start_snip <= j
#
#     df_ld_snps = load_snips_df(dir)
#
#     i_temp = df_ld_snps[df_ld_snps.BP == i]
#     j_temp = df_ld_snps[df_ld_snps.BP == j]
#     if len(i_temp) == 0 or len(j_temp) == 0:
#         return None
#     i_pos = i_temp.relative_pos[0]
#     j_pos = j_temp.relative_pos[0]
#     i_offset = i_pos % block_size
#     j_offset = j_pos % block_size
#
#     i_block = (i_pos // block_size) * block_size
#
#     # naively load everything you might possibly need
#     block_matrix = sp.load_npz(f'{dir}/block_{i_block}.npz')
#     off_diagonal_matrix = sp.load_npz(f'{dir}/col_{i_block}.npz').todok()
#
#     if j_pos < i_block + block_size:
#         return block_matrix[i_offset, j_offset]
#     else:
#         return off_diagonal_matrix[i_offset, j_offset]




def get_containing_files(snips_df, block_size, i_list, j_list):
    i_blocks = snips_df[snips_df.BP.isin(i_list)].relative_pos.floordiv(block_size).unique()
    j_blocks = snips_df[snips_df.BP.isin(j_list)].relative_pos.floordiv(block_size).unique()

    files = set()
    for i in i_blocks:
        for j in j_blocks:
            if i == j:
                files.add(f'block_{i}')
            elif i < j:
                files.add(f'col_{i}')
    return list(files)


def get_row(dir, i):
    start_snip, end_snip, block_size = get_metadata(dir)

    df_ld_snps = load_snips_df(dir)

    i_temp = df_ld_snps[df_ld_snps.BP == i]
    if len(i_temp) == 0:
        return None
    i_pos = i_temp.relative_pos[0]
    i_block = (i_pos // block_size) * block_size
    i_offset = i_pos % block_size

    # naively load everything you might possibly need
    block_matrix = sp.load_npz(f'{dir}/block_{i_block}.npz')
    off_diagonal_matrix = sp.load_npz(f'{dir}/row_{i_block}.npz') # to csc?

    return pd.DataFrame(np.hstack((block_matrix[i_offset].todense(), off_diagonal_matrix[i_offset].todense())),
                        index=[df_ld_snps.index[i_pos]], columns=df_ld_snps.index)


def get_rows(dir, i_list): #need to transpose
    i_list = sorted(i_list)

    start_snip, end_snip, block_size = get_metadata(dir)

    df_ld_snps = load_snips_df(dir)

    i_temp = df_ld_snps[df_ld_snps.BP.isin(i_list)].relative_pos
    if len(i_temp) != len(i_list):
        return 'error' #needs handling

    i_blocks = i_temp.floordiv(block_size) * block_size
    i_offset = i_temp.mod(block_size)

    rows = []
    loaded_block = None
    loaded_block_matrix = None
    loaded_aux_matrix = None
    for block, offset in zip(i_blocks, i_offset):
        if loaded_block != block:
            loaded_block = block
            loaded_block_matrix = sp.load_npz(f'{dir}/block_{block}.npz')
            if os.path.exists(f'{dir}/row_{block}.npz'): #do this logically, no file lookup
                loaded_aux_matrix = sp.load_npz(f'{dir}/row_{block}.npz')
            else:
                loaded_aux_matrix = None

        row = [[np.zeros(block)], loaded_block_matrix[offset].todense()]


        if loaded_aux_matrix is not None:
            row.append(loaded_aux_matrix[offset].todense())

        rows.append(np.hstack(row))

    return pd.DataFrame(np.vstack(rows), index = df_ld_snps.iloc[i_temp].index, columns=df_ld_snps.index)


def get_submatrix_v1(dir, i_list, j_list):
    i_list = sorted(i_list)
    j_list = sorted(j_list)

    start_snip, end_snip, block_size = get_metadata(dir)

    df_ld_snps = load_snips_df(dir)

    i_temp = df_ld_snps[df_ld_snps.BP.isin(i_list)].relative_pos
    if len(i_temp) != len(i_list):
        return 'error' #needs handling

    i_blocks = i_temp.floordiv(block_size) * block_size
    i_offset = i_temp.mod(block_size)

    df_ld_snps = load_snips_df(dir)

    j_temp = df_ld_snps[df_ld_snps.BP.isin(j_list)].relative_pos
    if len(j_temp) != len(j_list):
        return 'error' #needs handling

    j_blocks = j_temp.floordiv(block_size) * block_size
    j_offset = j_temp.mod(block_size)

    #size of final table is len(ilist) x len(jlist)

    rows = []


    for block in range(0, end_snip - start_snip, block_size):
        local_is = i_temp[i_blocks == block] #value should be relative_pos
        local_js = j_temp[j_blocks == block]

        if len(local_is) == 0 and len(local_js) == 0:
            continue

        local_i_offsets = list(local_is.mod(block_size))
        local_j_offsets = list(local_js.mod(block_size))

        row = []
        aux_matrix = None

        if len(local_js) and len(local_is) < len(i_list):
            aux_matrix = sp.load_npz(f'{dir}/row_{block}.npz')
            nonlocal_is = i_temp[i_temp < block] # think about this
            row.append(aux_matrix[np.ix_(local_j_offsets, nonlocal_is)].todense())
        else:
            pass # append zeros?

        if len(local_is) and len(local_js):
            block_matrix = sp.load_npz(f'{dir}/block_{block}.npz')
            row.append(block_matrix[np.ix_(local_i_offsets, local_j_offsets)].todense())
        else:
            pass #append zeros?


        if len(local_is) and len(local_js) < len(j_list):
            if aux_matrix is None:
                aux_matrix = sp.load_npz(f'{dir}/row_{block}.npz')
            # find all js greater than blocksize + block
            nonlocal_js = j_temp[j_temp > (block + block_size)]
            row.append(aux_matrix[np.ix_(local_i_offsets, nonlocal_js)].todense())
        else:
            # append zeros?
            pass

        # may need to fill zeros to reach len(jlist)
        # first take all the js that come before the block
        # then all the js that are in the block
        # then all js that are after the block

        rows.append(np.hstack(row))

    return pd.DataFrame(np.vstack(rows), index=df_ld_snps.iloc[i_temp].index, columns=df_ld_snps.iloc[j_temp].index)

def get_submatrix_v2(dir, i_list, j_list):
    i_list = sorted(i_list)
    j_list = sorted(j_list)

    start_snip, end_snip, block_size = get_metadata(dir)

    df_ld_snps = load_snips_df(dir)

    i_temp = df_ld_snps[df_ld_snps.BP.isin(i_list)].relative_pos
    if len(i_temp) != len(i_list):
        return 'error'  # needs handling

    i_blocks = i_temp.floordiv(block_size) * block_size
    i_offset = i_temp.mod(block_size)

    df_ld_snps = load_snips_df(dir)

    j_temp = df_ld_snps[df_ld_snps.BP.isin(j_list)].relative_pos
    if len(j_temp) != len(j_list):
        return 'error'  # needs handling

    j_blocks = j_temp.floordiv(block_size) * block_size
    j_offset = j_temp.mod(block_size)

    # size of final table is len(ilist) x len(jlist)

    rows = []

    for block in range(0, end_snip - start_snip, block_size):
        local_is = i_temp[i_blocks == block]  # value should be relative_pos
        local_js = j_temp[j_blocks == block]

        if len(local_is) == 0 and len(local_js) == 0:
            continue

        local_i_offsets = list(local_is.mod(block_size))
        local_j_offsets = list(local_js.mod(block_size))

        nonlocal_is = i_temp[i_temp > block + block_size]  # think about this
        nonlocal_js = j_temp[j_temp > block + block_size]

        row = [np.zeros((len(set(local_i_offsets + local_j_offsets)), len(i_temp[i_temp < block])))]
        aux_matrix = None
        #
        # if len(local_js) and len(local_is) < len(i_list):
        #     aux_matrix = sp.load_npz(f'{dir}/row_{block}.npz')
        #     row.append(aux_matrix[np.ix_(local_j_offsets, nonlocal_is)].todense())
        # else:
        #     pass  # append zeros?

        if len(local_is) and len(local_js):
            block_matrix = sp.load_npz(f'{dir}/block_{block}.npz')
            row.append(block_matrix[np.ix_(local_i_offsets, local_j_offsets)].todense())
        else:
            pass  # append zeros?

        if len(local_is) and len(local_js) < len(j_list):
            if aux_matrix is None:
                aux_matrix = sp.load_npz(f'{dir}/row_{block}.npz')
            # find all js greater than blocksize + block
            row.append(aux_matrix[np.ix_(sorted(set(local_i_offsets + local_j_offsets)), sorted(set(list(nonlocal_js) + list(nonlocal_is))))].todense())
        else:
            # append zeros?
            pass

        # may need to fill zeros to reach len(jlist)
        # first take all the js that come before the block
        # then all the js that are in the block
        # then all js that are after the block
        rows.append(np.hstack(row))

    breakpoint()
    return pd.DataFrame(np.vstack(rows), index=df_ld_snps.iloc[i_temp].index, columns=df_ld_snps.iloc[j_temp].index)

    #iterate through all possible blocks + offdiagonals
    # if its needed, load and query all


def get_submatrix_fresh(dir, i_list, j_list):
    i_list = sorted(i_list)
    j_list = sorted(j_list)

    start_snip, end_snip, block_size = get_metadata(dir)

    df_ld_snps = load_snips_df(dir)

    i_temp = df_ld_snps[df_ld_snps.BP.isin(i_list)].relative_pos
    if len(i_temp) != len(i_list):
        return 'error'  # needs handling

    i_blocks = i_temp.floordiv(block_size) * block_size

    df_ld_snps = load_snips_df(dir)

    j_temp = df_ld_snps[df_ld_snps.BP.isin(j_list)].relative_pos
    if len(j_temp) != len(j_list):
        return 'error'  # needs handling

    j_blocks = j_temp.floordiv(block_size) * block_size

    # size of final table is len(ilist) x len(jlist)

    rows = []

    for block in range(0, end_snip - start_snip, block_size):
        local_is = i_temp[i_blocks == block]  # value should be relative_pos
        local_js = j_temp[j_blocks == block]

        early_is = i_temp[i_blocks < block]
        late_is = i_temp[i_blocks > block]
        leading_zero_width = len(early_is)

        if len(local_is) == 0 and len(local_js) == 0:
            continue

        local_i_offsets = list(local_is.mod(block_size))
        local_j_offsets = list(local_js.mod(block_size))
        late_i_offsets = list(late_is.mod(block_size))

        # section has height local_js
        # section has width i_list
        row = [np.zeros((len(local_js), leading_zero_width))]


        if len(local_is) and len(local_js):
            block_matrix = sp.load_npz(f'{dir}/block_{block}.npz')
            row.append(block_matrix[np.ix_(local_j_offsets, local_i_offsets)].todense())

        if len(local_js) and len(late_is):
            aux_matrix = sp.load_npz(f'{dir}/row_{block}.npz')
            row.append(aux_matrix[np.ix_(local_j_offsets, late_is - (block + block_size))].todense())

        #REMOVE
        # row.append(np.zeros((len(local_js), len(late_is))))
        # breakpoint()
        rows.append(np.hstack(row))

    # breakpoint()

    df = pd.DataFrame(np.vstack(rows), index=df_ld_snps.iloc[j_temp].index, columns=df_ld_snps.iloc[i_temp].index)
    return make_symmetric(df)

def make_symmetric(A):
    #inefficient
    for i in A.index:
        for j in A.columns:
            if i in A and j in A[i]:
                if A[i][j] != 0:
                    A[j][i] = A[i][j]
                else:
                    A[i][j] = A[j][i]
    return A


def get_submatrix_by_indices(dir, i_list, j_list):
    i_list = sorted(i_list)
    j_list = sorted(j_list)

    ind_list = sorted(set(i_list + j_list))

    start_snip, end_snip, block_size = get_metadata(dir)

    df_ld_snps = load_snips_df(dir)

    ind_temp = df_ld_snps[df_ld_snps.BP.isin(ind_list)].relative_pos

    ind_blocks = ind_temp.floordiv(block_size) * block_size

    rows = []

    for block in range(0, end_snip - start_snip, block_size):
        early_inds = ind_temp[ind_blocks < block]
        local_inds = ind_temp[ind_blocks == block]
        late_inds = ind_temp[ind_blocks > block]

        if len(local_inds) == 0:
            continue

        local_ind_offsets = list(local_inds.mod(block_size))

        row = [np.zeros((len(local_inds), len(early_inds)))]

        block_matrix = sp.load_npz(f'{dir}/block_{block}.npz') # could be avoided when selecting non overlapping
        row.append(block_matrix[np.ix_(local_ind_offsets, local_ind_offsets)].todense())


        if len(late_inds):
            aux_matrix = sp.load_npz(f'{dir}/row_{block}.npz')
            row.append(aux_matrix[np.ix_(local_ind_offsets, late_inds - (block + block_size))].todense())

        rows.append(np.hstack(row))

    triangular = np.vstack(rows)
    full = triangular + triangular.T

    i_temp = df_ld_snps[df_ld_snps.BP.isin(i_list)].relative_pos
    j_temp = df_ld_snps[df_ld_snps.BP.isin(j_list)].relative_pos

    ld_snps_ind = df_ld_snps.iloc[ind_temp].index

    # should reduce size before creating DF
    df = pd.DataFrame(full, index=ld_snps_ind, columns =ld_snps_ind)
    return df.loc[df_ld_snps.iloc[j_temp].index, df_ld_snps.iloc[i_temp].index]


def get_submatrix_by_ranges(dir, i_start, i_end, j_start, j_end):
    return get_submatrix_by_indices(dir, range(i_start, i_end), range(j_start, j_end))


#maybe make an object






# convert('data/chr1_194000001_197000001', 'data/chr1_194000001_197000001', 2000, .1)

dir = 'data/chr1_194000001_197000001_2000_0.1'
testrows = [194000205, 194000389, 194000398, 194021357, 194252729, 194806501, 195267072]
limited_testrows = [194000205, 194000389, 194000398, 194021357, 194252729, 195267072]

# submat = get_submatrix_fresh(dir, testrows, testrows)
# row_only = make_symmetric(get_rows(dir, testrows)[submat.columns])
#
# submat2 = get_submatrix_fresh(dir, limited_testrows, limited_testrows)
# reducedmat = get_submatrix_fresh(dir, limited_testrows, testrows)
#
# row_only = make_symmetric(get_rows(dir, testrows)[submat.columns])[reducedmat.columns]
# final = get_submatrix_simple(dir, limited_testrows, testrows)