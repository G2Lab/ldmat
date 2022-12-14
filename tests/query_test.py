import filecmp
from tempfile import NamedTemporaryFile

import numpy as np
import scipy.sparse as sp
from click.testing import CliRunner

from src.ldmat.__main__ import submatrix, submatrix_by_maf


def test_npz_query():
    with NamedTemporaryFile(suffix=".npz") as tmp:

        runner = CliRunner()
        result = runner.invoke(
            submatrix,
            [
                "examples/chr21_partial.h5",
                "--row-start",
                14900001,
                "--row-end",
                15100001,
                "--outfile",
                tmp.name,
            ],
        )
        assert result.exit_code == 0, print(result.output)

        assert filecmp.cmp(tmp.name, "examples/query_result.npz")


def test_csv_query():
    with NamedTemporaryFile(suffix=".csv") as tmp:

        runner = CliRunner()
        result = runner.invoke(
            submatrix,
            [
                "examples/chr21_partial.h5",
                "--row-start",
                14900001,
                "--row-end",
                15100001,
                "--outfile",
                tmp.name,
            ],
        )
        assert result.exit_code == 0, print(result.output)

        assert filecmp.cmp(tmp.name, "examples/query_result.csv")


def test_maf_query():
    with NamedTemporaryFile(suffix=".npz") as tmp:

        runner = CliRunner()
        result = runner.invoke(
            submatrix_by_maf,
            [
                "examples/chr21_partial.h5",
                "--lower-bound",
                0.2,
                "--upper-bound",
                0.22,
                "--outfile",
                tmp.name,
            ],
        )
        assert result.exit_code == 0, print(result.output)

        assert filecmp.cmp(tmp.name, "examples/query_result_maf.npz")


def test_streaming_query():
    with NamedTemporaryFile(suffix=".csv") as tmp:

        runner = CliRunner()
        result = runner.invoke(
            submatrix,
            [
                "examples/chr21_partial.h5",
                "--row-start",
                14900001,
                "--row-end",
                15100001,
                "--outfile",
                tmp.name,
                "--stream",
            ],
        )
        assert result.exit_code == 0, print(result.output)

        assert filecmp.cmp(tmp.name, "examples/query_result.csv")


def test_streaming_maf_query():
    with NamedTemporaryFile(suffix=".npz") as tmp:

        runner = CliRunner()
        result = runner.invoke(
            submatrix_by_maf,
            [
                "examples/chr21_partial.h5",
                "--lower-bound",
                0.2,
                "--upper-bound",
                0.22,
                "--outfile",
                tmp.name,
                "--stream",
            ],
        )
        assert result.exit_code == 0, print(result.output)

        test_output = np.nan_to_num(sp.load_npz(tmp.name).todense())
        correct_output = np.nan_to_num(
            sp.load_npz("examples/query_result_maf.npz").todense()
        )

        assert np.abs(test_output - correct_output).max() < 0.00001
