import numpy as np
import scipy.sparse as sparse
import os
import pandas as pd
import click
import h5py
from heapq import merge

DIAGONAL_LD = 1
FULL_MATRIX_NAME = "full"

MAF_COLS = [
    "Alternate_id",
    "RS_id",
    "Position",
    "Allele1",
    "Allele2",
    "MAF",
    "Minor Allele",
    "Info Score",
]


def unique_merge(v):
    last = None
    for a in merge(*v):
        if a != last:
            last = a
            yield a


def adjust_to_zero(sparse_matrix, precision):
    if precision:
        nonzeros = sparse_matrix.nonzero()
        nonzero_mask = np.array(np.abs(sparse_matrix[nonzeros]) < precision)[0]
        rows = nonzeros[0][nonzero_mask]
        cols = nonzeros[1][nonzero_mask]
        sparse_matrix[rows, cols] = 0

    sparse_matrix.eliminate_zeros()
    return sparse_matrix


def convert_h5(
    infile, outfile, precision=0, decimals=3, start_snip=None, end_snip=None
):
    base_infile = os.path.splitext(infile)[0]

    if not start_snip or not end_snip:
        filename = base_infile.split("/")[-1]
        chromosome, start_snip, end_snip = filename.split("_")
        start_snip, end_snip = int(start_snip), int(end_snip)

    print(f"Converting {infile} snips {start_snip} to {end_snip}")

    f = h5py.File(outfile, "a")

    group = f.require_group(f"snip_{start_snip}")

    sparse_mat = sparse.triu(sparse.load_npz(base_infile + ".npz").T, format="csr")
    sparse_mat.setdiag(0)
    sparse_mat.data = np.round(sparse_mat.data, decimals)
    sparse_mat = adjust_to_zero(sparse_mat, precision)

    pos_df = metadata_to_df(base_infile + ".gz")
    group.require_dataset(
        "positions",
        data=pos_df,
        compression="gzip",
        shape=pos_df.shape,
        dtype=pos_df.dtypes[0],
    )
    names = pos_df.index.to_numpy().astype("S")
    group.require_dataset(
        "names", data=names, shape=names.shape, dtype=names.dtype, compression="gzip"
    )

    group.attrs["start_snip"] = start_snip
    group.attrs["end_snip"] = end_snip
    group.attrs["precision"] = precision

    pos_df["relative_pos"] = np.arange(len(pos_df))
    # actually should not filter, since need for rows. instead save start and end snips for columns
    pos_df = pos_df[pos_df.BP.between(start_snip, end_snip)]

    if len(pos_df) == 0:
        print(f"No data found between snips {start_snip} and {end_snip}")
        return None

    lower_pos, upper_pos = pos_df.relative_pos[[0, -1]]
    sparse_mat = sparse_mat[lower_pos : upper_pos + 1, :]
    dense = sparse_mat.todense()
    group.require_dataset(
        "full",
        data=dense,
        compression="gzip",
        compression_opts=9,
        shape=dense.shape,
        dtype=dense.dtype,
        scaleoffset=decimals,
    )


def extract_metadata_df_from_group(group):
    df = pd.DataFrame(group["positions"], columns=["BP"], index=group["names"])
    df["relative_pos"] = np.arange(len(df))
    return df


def metadata_to_df(gz_file):
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
    df_ld_snps.index = (
        df_ld_snps["CHR"].astype(str)
        + "."
        + df_ld_snps["BP"].astype(str)
        + "."
        + df_ld_snps["A1"]
        + "."
        + df_ld_snps["A2"]
    )
    return df_ld_snps[["BP"]]


def add_slice_to_df(df, slice, axis):
    if slice.empty:
        return df
    if df is None:
        return slice
    if df.axes[axis][-1] not in slice.axes[axis]:
        return pd.concat((df, slice), axis=axis)

    if axis == 0:
        infill = slice.loc[: df.index[-1], :]
        extension = slice.loc[df.index[-1] :, :].iloc[1:, :]
    else:
        infill = slice.loc[:, : df.columns[-1]]
        extension = slice.loc[:, df.columns[-1] :].iloc[:, 1:]

    df.loc[
        infill.index[0] : infill.index[-1], infill.columns[0] : infill.columns[-1]
    ] = infill
    return pd.concat((df, extension), axis=axis)


def add_main_slice_to_df(df, main_slice):
    if df is None:
        return main_slice
    if main_slice.index[0] not in df.index:
        return pd.concat((df, main_slice), axis=0)

    if main_slice.index[-1] in df.index:
        # everything is normal
        df.loc[
            main_slice.index[0] : main_slice.index[-1],
            main_slice.columns[0] : main_slice.columns[-1],
        ] = main_slice
    else:
        # main slice goes too far
        df.loc[main_slice.index[0] :, main_slice.columns[0] :] = main_slice.loc[
            : df.index[-1], : df.index[-1]
        ]
        df = pd.concat(
            (df, main_slice.loc[df.index[-1] :, df.index[-1] :].iloc[1:, 1:]), axis=0
        )

    return df


def subselect(df, rows, columns, range_query):
    BP_list = df[[]].copy()
    BP_list["BP"] = df.index.str.split(".").str[1].astype(int)
    BP_list["relative_pos"] = np.arange(len(BP_list))

    if range_query:
        r = BP_list[BP_list.BP.between(*rows)]
        c = BP_list[BP_list.BP.between(*columns)]
    else:
        r = BP_list[BP_list.BP.isin(rows)]
        c = BP_list[BP_list.BP.isin(columns)]

    if len(r) == 0 or len(c) == 0:
        return pd.DataFrame()

    row_inds, col_inds = r.relative_pos, c.relative_pos

    if range_query:
        return df.iloc[row_inds[0] : row_inds[-1] + 1, col_inds[0] : col_inds[-1] + 1]
    else:
        return df.iloc[row_inds, col_inds]


def get_horizontal_slice(group, rows, columns, range_query):
    df_ld_snps = extract_metadata_df_from_group(group)
    if range_query:
        r = df_ld_snps.BP.between(*rows)
        c = df_ld_snps.BP.between(*columns)
    else:
        r = df_ld_snps.BP.isin(rows)
        c = df_ld_snps.BP.isin(rows)

    row_positions = df_ld_snps[r].relative_pos
    col_positions = df_ld_snps[c].relative_pos

    if len(row_positions) and len(col_positions):
        if range_query:
            slice = group[FULL_MATRIX_NAME][
                row_positions[0] : row_positions[-1] + 1,
                col_positions[0] : col_positions[-1] + 1,
            ]
        else:
            slice = group[FULL_MATRIX_NAME][row_positions.tolist()][:, col_positions]
    else:
        slice = None

    return pd.DataFrame(
        slice,
        index=row_positions.index.astype(str),
        columns=col_positions.index.astype(str),
    )


def overlap(a, b, range_query):
    if not range_query:
        return [index for index in a if b[0] <= index < b[1]]
    overlap = max(a[0], b[0]), min(a[1], b[1])
    if overlap[1] < overlap[0]:
        return None
    return overlap


def outer_overlap(i_overlap, j_overlap, range_query):
    merged = list(unique_merge((i_overlap, j_overlap)))
    return merged if range_query else merged[0], merged[-1]


def get_submatrix_from_chromosome(chromosome_group, i_values, j_values, range_query):
    if (
        len(i_values) == 0
        or len(j_values) == 0
        or i_values[-1] < i_values[0]
        or j_values[-1] < j_values[0]
    ):
        return pd.DataFrame()

    i_values, j_values = sorted(set(i_values)), sorted(set(j_values))

    intervals = []
    for subgroup in chromosome_group.values():
        if subgroup.name != "/aux":
            intervals.append((subgroup.attrs["start_snip"], subgroup.attrs["end_snip"]))

    intervals.sort(key=lambda x: x[0])

    df = None
    for interval in intervals:
        # INTERVALS ARE ONLY FOR i VALUES!!!

        i_overlap = overlap(i_values, interval, range_query)
        j_overlap = overlap(j_values, interval, range_query)

        group = chromosome_group[f"snip_{interval[0]}"]

        # get right overlap, bottom overlap, triangle overlap
        if i_overlap and j_overlap:
            full_overlap = outer_overlap(i_overlap, j_overlap, range_query)

            # triangular slice - full triangle, make symmetric, then subselect - can probably be done more efficiently
            main_slice = get_horizontal_slice(
                group, full_overlap, full_overlap, range_query
            )
            if not main_slice.empty:
                main_slice = main_slice + main_slice.T
                np.fill_diagonal(main_slice.values, DIAGONAL_LD)
                main_slice = subselect(main_slice, i_overlap, j_overlap, range_query)

                df = add_main_slice_to_df(df, main_slice)

            # right slice - all i in overlap, all j > interval end
        if i_overlap and j_values[-1] > interval[1]:
            right_slice = get_horizontal_slice(
                group,
                i_overlap,
                overlap(j_values, (interval[1], np.inf), range_query),
                range_query,
            )
            df = add_slice_to_df(df, right_slice, 1)

        # bottom slice - all j in overlap, all i > interval end
        if j_overlap and i_values[-1] > interval[1]:
            bottom_slice = get_horizontal_slice(
                group,
                j_overlap,
                overlap(i_values, (interval[1], np.inf), range_query),
                range_query,
            ).T
            df = add_slice_to_df(df, bottom_slice, 0)

    return df.fillna(0)


def get_maf_indices_by_range(maf_dataset, lower_bound, upper_bound):
    # inclusive
    maf_values = maf_dataset[:, 1]
    start_index = np.searchsorted(maf_values, lower_bound)
    while start_index > 0 and maf_values[start_index - 1] >= lower_bound:
        start_index -= 1

    end_index = np.searchsorted(maf_values, upper_bound)
    while end_index < len(maf_values) and maf_values[end_index + 1] <= upper_bound:
        end_index += 1

    return maf_dataset[start_index:end_index, 0]


def get_submatrix_by_maf_range(chromosome_group, lower_bound, upper_bound):
    indices = get_maf_indices_by_range(
        chromosome_group["aux"]["MAF"], lower_bound, upper_bound
    )
    print(f"Found {len(indices)} matching MAFs")
    maf_result = get_submatrix_from_chromosome(
        chromosome_group, indices, indices, range_query=False
    )
    assert all(np.diagonal(maf_result) == 1)
    return maf_result


def convert_maf_h5(infile, outfile):
    f = h5py.File(outfile, "a")
    group = f.require_group("aux")

    maf = pd.read_csv(infile, sep="\t", header=None, names=MAF_COLS)
    maf = maf.sort_values("MAF")
    maf = maf[["Position", "MAF"]].to_numpy()
    group.require_dataset(
        "MAF", data=maf, shape=maf.shape, dtype=maf.dtype, compression="gzip"
    )


# maybe make an object


@click.group()
def cli():
    pass


@cli.command()
@click.argument("ld_file")
@click.option("--i_start", type=int)
@click.option("--i_end", type=int)
@click.option("--j_start", type=int)
@click.option("--j_end", type=int)
@click.option("--outfile", "-o", default=None)
@click.option("--symmetric", "-s", is_flag=True, default=False)
def submatrix(ld_file, i_start, i_end, j_start, j_end, outfile, symmetric):
    if symmetric and (j_start is not None or j_end is not None):
        raise ValueError("Symmetric flag only compatible with i indexing.")
    if symmetric:
        j_start, j_end = i_start, i_end
    res = get_submatrix_from_chromosome(
        h5py.File(ld_file, "r"), (i_start, i_end), (j_start, j_end), range_query=True
    )
    if outfile:
        # name index?
        res.to_csv(outfile)
    else:
        print(res)


@cli.command()
@click.argument("infile", type=click.Path())
@click.argument("outfile", type=click.Path(exists=False))
@click.option("--precision", "-p", type=float, default=0)
@click.option("--decimals", "-d", type=int, default=3)
@click.option("--start_snip", "-s", type=int, default=None)
@click.option("--end_snip", "-e", type=int, default=None)
def convert(infile, outfile, precision, decimals, start_snip, end_snip):
    convert_h5(infile, outfile, precision, decimals, start_snip, end_snip)


@cli.command()
@click.argument("infile", type=click.Path())
@click.argument("outfile", type=click.Path(exists=False))
def convert_maf(infile, outfile):
    convert_maf_h5(infile, outfile)


@cli.command()
@click.argument("ld_file")
@click.option("--lower_bound", "-l", type=float, default=0)
@click.option("--upper_bound", "-u", type=float, default=0.5)
@click.option("--outfile", "-o", default=None)
def submatrix_by_maf(ld_file, lower_bound, upper_bound, outfile):
    res = get_submatrix_by_maf_range(h5py.File(ld_file, "r"), lower_bound, upper_bound)
    if outfile:
        # name index?
        res.to_csv(outfile)
    else:
        print(res)


@cli.command()
@click.argument("directory", type=click.Path())
@click.argument("chromosome", type=int)
@click.argument("outfile", type=click.Path(exists=False))
@click.option("--precision", "-p", type=float, default=0)
@click.option("--decimals", "-d", type=int, default=3)
@click.option("--start_snip", "-s", type=int, default=1)
def convert_chromosome(directory, chromosome, outfile, precision, decimals, start_snip):
    print(f"Converting chromosome {chromosome}")

    filtered = []
    for file in os.listdir(directory):
        if (
            os.path.isfile(os.path.join(directory, file))
            and file.startswith(f"chr{chromosome}_")
            and file.endswith(".npz")
        ):
            filtered.append((file, int(file.split("_")[1])))

    filtered.sort(key=lambda x: x[1])

    start_snip = max(start_snip, filtered[0][1])

    first_missing_snip = start_snip

    for i, (file, snip) in enumerate(filtered):
        if snip >= start_snip:
            print(f"Converting {file}")
            if i + 1 < len(filtered):
                next_covered_snip = filtered[i + 1][1]
            else:
                next_covered_snip = np.inf
            convert_h5(
                os.path.join(directory, file),
                outfile,
                precision,
                decimals,
                first_missing_snip,
                next_covered_snip,
            )
            first_missing_snip = next_covered_snip


if __name__ == "__main__":
    cli()
