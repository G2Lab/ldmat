import numpy as np
import scipy.sparse as sparse
import scipy.linalg as linalg
import os
import pandas as pd
import click
import shutil
from heapq import merge


def sort_and_combine_lists(a, b):
    sorted_a, sorted_b = sorted(set(a)), sorted(set(b))
    return sorted_a, sorted_b, list(merge(sorted_a, sorted_b))


def combine_matrices(matrices):
    columns = [column for matrix in matrices for column in matrix.columns]
    index = [index for matrix in matrices for index in matrix.index]
    return pd.DataFrame(linalg.block_diag(*matrices), columns=columns, index=index)


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


def convert(infile, outdir, block_size, precision=0):
    base_infile = os.path.splitext(infile)[0]
    filename = base_infile.split("/")[-1]
    chromosome, start_snip, end_snip = filename.split("_")
    start_snip, end_snip = int(start_snip), int(end_snip)
    chromosome_dir = os.path.join(outdir, chromosome + f"_{precision}")
    if not os.path.exists(chromosome_dir):
        os.makedirs(chromosome_dir)
    outdir = os.path.join(chromosome_dir, f"{start_snip}_{end_snip}")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    sparse_mat = sparse.triu(sparse.load_npz(base_infile + ".npz").T, format="csr")
    mat_size = sparse_mat.shape[0]

    shutil.copy(base_infile + ".gz", os.path.join(outdir, "metadata.gz"))

    pd.DataFrame(
        [[start_snip, end_snip, block_size, precision]],
        columns=["start_snip", "end_snip", "block_size", "precision"],
    ).to_csv(os.path.join(outdir, "META"), index=False)

    for i in range(0, mat_size, block_size):
        if i + block_size < mat_size:
            off_diagonal = sparse_mat[i : i + block_size, i + block_size :]

            off_diagonal = adjust_to_zero(
                off_diagonal, precision
            )  # or is tocoo() better?
            print(f"Writing off diagonal {i}")
            sparse.save_npz(
                os.path.join(outdir, f"row_{i}"), off_diagonal, compressed=True
            )
            print("Finished writing")

        reduced = reduce_submatrix(
            sparse_mat, i, i + block_size, precision
        )  # allow precision?
        print(f"Writing block {i}")
        sparse.save_npz(os.path.join(outdir, f"block_{i}"), reduced, compressed=True)
        print("Finished writing")


def get_metadata(dir):
    return pd.read_csv(dir + "/META").iloc[0]


def load_snips_df(dir):
    # load the SNPs metadata
    gz_file = dir + "/metadata.gz"
    df_ld_snps = pd.read_table(gz_file, sep="\s+")
    df_ld_snps.rename(
        columns={
            "rsid": "SNP",
            "chromosome": "CHR",
            "position": "BP",
            "allele1": "A1",
            "allele2": "A2",
        },
        inplace=True,
        errors="ignore",
    )
    assert "SNP" in df_ld_snps.columns
    assert "CHR" in df_ld_snps.columns
    assert "BP" in df_ld_snps.columns
    assert "A1" in df_ld_snps.columns
    assert "A2" in df_ld_snps.columns
    df_ld_snps.rename_axis("relative_pos", inplace=True)
    df_ld_snps.reset_index(inplace=True)
    df_ld_snps.index = (
        df_ld_snps["CHR"].astype(str)
        + "."
        + df_ld_snps["BP"].astype(str)
        + "."
        + df_ld_snps["A1"]
        + "."
        + df_ld_snps["A2"]
    )
    return df_ld_snps


def find_interval_index(val, intervals, start_index):
    index = start_index
    start_snip, end_snip = intervals[start_index]
    while val >= end_snip:
        index += 1
        if index == len(intervals):
            return None
        start_snip, end_snip = intervals[index]
    return index


def find_overlap(a, b):
    overlap = max(a[0], b[0]), min(a[1], b[1])
    if overlap[1] < overlap[0]:
        return None
    return overlap


def get_submatrix_from_chromosome_by_range(
    chromosome_dir, i_start, i_end, j_start, j_end
):
    intervals = []
    for subdir in [x[0] for x in os.walk(chromosome_dir)][1:]:
        dirname = subdir.split("/")[-1]
        start, end = dirname.split("_")
        intervals.append((int(start), int(end)))

    intervals.sort(key=lambda x: x[0])

    submatrices = []
    for interval in intervals:
        i_overlap = find_overlap((i_start, i_end), interval)
        j_overlap = find_overlap((j_start, j_end), interval)
        if i_overlap or j_overlap:
            i_overlap = i_overlap or (-1, -1)
            j_overlap = j_overlap or (-1, -1)
            submatrices.append(
                get_submatrix_by_ranges(
                    os.path.join(chromosome_dir, f"{interval[0]}_{interval[1]}"),
                    *i_overlap,
                    *j_overlap,
                )
            )
    return combine_matrices(submatrices)


def get_submatrix_from_chromosome(chromosome_dir, i_list, j_list):
    i_list, j_list, ind_list = sort_and_combine_lists(i_list, j_list)

    # need to find all intervals and compare

    intervals = []
    for subdir in [x[0] for x in os.walk(chromosome_dir)][1:]:
        dirname = subdir.split("/")[-1]
        start, end = dirname.split("_")
        intervals.append((int(start), int(end)))

    intervals.sort(key=lambda x: x[0])

    submatrices = []
    prev = 0
    interval_ind = find_interval_index(ind_list[0], intervals, 0)
    start_snip, end_snip = intervals[interval_ind]
    for i, ind in enumerate(ind_list):
        if ind >= end_snip:
            submatrices.append(
                get_submatrix_by_indices(
                    os.path.join(chromosome_dir, f"{start_snip}_{end_snip}"),
                    ind_list[prev:i],
                    ind_list[prev:i],
                )
            )
            prev = i
            interval_ind = find_interval_index(ind, intervals, interval_ind)
            start_snip, end_snip = intervals[interval_ind]

    submatrices.append(
        get_submatrix_by_indices(
            os.path.join(chromosome_dir, f"{start_snip}_{end_snip}"),
            ind_list[prev:],
            ind_list[prev:],
        )
    )
    return combine_matrices(submatrices)


def load_symmetric_matrix(dir, index_df):
    metadata = get_metadata(dir)
    start_snip, end_snip, block_size = (
        metadata.start_snip,
        metadata.end_snip,
        metadata.block_size,
    )

    ind_blocks = index_df.floordiv(block_size) * block_size

    rows = []

    for block in range(0, end_snip - start_snip, block_size):
        early_inds = index_df[ind_blocks < block]
        local_inds = index_df[ind_blocks == block]
        late_inds = index_df[ind_blocks > block]

        if len(local_inds) == 0:
            continue

        local_ind_offsets = list(local_inds.mod(block_size))

        row = [np.zeros((len(local_inds), len(early_inds)))]

        block_matrix = sparse.load_npz(
            f"{dir}/block_{block}.npz"
        )  # could be avoided when selecting non overlapping
        row.append(block_matrix[np.ix_(local_ind_offsets, local_ind_offsets)].todense())

        if len(late_inds):
            aux_matrix = sparse.load_npz(f"{dir}/row_{block}.npz")
            row.append(
                aux_matrix[
                    np.ix_(local_ind_offsets, late_inds - (block + block_size))
                ].todense()
            )

        rows.append(np.hstack(row))

    if not len(rows):
        return np.empty((0, 0))
    triangular = np.vstack(rows)
    return triangular + triangular.T


def construct_labeled_df(
    full_matrix, df_ld_snps, i_index_df, j_index_df, combined_index_df
):
    ld_snps_ind = df_ld_snps.iloc[combined_index_df].index

    # should reduce size before creating DF
    df = pd.DataFrame(full_matrix, index=ld_snps_ind, columns=ld_snps_ind)
    return df.loc[df_ld_snps.iloc[j_index_df].index, df_ld_snps.iloc[i_index_df].index]


def get_submatrix_by_indices(dir, i_list, j_list):
    i_list, j_list, ind_list = sort_and_combine_lists(i_list, j_list)

    df_ld_snps = load_snips_df(dir)

    ind_temp = df_ld_snps[df_ld_snps.BP.isin(ind_list)].relative_pos

    i_temp = df_ld_snps[df_ld_snps.BP.isin(i_list)].relative_pos
    j_temp = df_ld_snps[df_ld_snps.BP.isin(j_list)].relative_pos

    return construct_labeled_df(
        load_symmetric_matrix(dir, ind_temp), df_ld_snps, i_temp, j_temp, ind_temp
    )


def get_submatrix_by_ranges(dir, i_start, i_end, j_start, j_end):
    df_ld_snps = load_snips_df(dir)

    ind_temp = df_ld_snps[
        df_ld_snps.BP.between(i_start, i_end) | df_ld_snps.BP.between(j_start, j_end)
    ].relative_pos

    i_temp = df_ld_snps[df_ld_snps.BP.between(i_start, i_end)].relative_pos
    j_temp = df_ld_snps[df_ld_snps.BP.between(j_start, j_end)].relative_pos

    return construct_labeled_df(
        load_symmetric_matrix(dir, ind_temp), df_ld_snps, i_temp, j_temp, ind_temp
    )


def get_value_at_index(dir, i, j):
    return get_submatrix_by_indices(dir, [i], [j])


# maybe make an object


@click.command()
@click.argument("chromosome_dir")
@click.option("--i_start", type=int)
@click.option("--i_end", type=int)
@click.option("--j_start", type=int)
@click.option("--j_end", type=int)
@click.option("--outfile", "-o", default=None)
@click.option("--symmetric", "-s", is_flag=True, default=False)
def submatrix(chromosome_dir, i_start, i_end, j_start, j_end, outfile, symmetric):
    if symmetric and (j_start is not None or j_end is not None):
        raise ValueError("Symmetric flag only compatible with i indexing.")
    if symmetric:
        j_start, j_end = i_start, i_end
    res = get_submatrix_from_chromosome_by_range(
        chromosome_dir, i_start, i_end, j_start, j_end
    )
    if outfile:
        res.to_csv(outfile)
    else:
        print(res)


if __name__ == "__main__":
    submatrix()

# example: python3 main.py data/processed/chr1_0 --i_start 10000 --i_end 100000 -s -o /tmp/test.csv
