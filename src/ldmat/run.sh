#!/bin/bash
#SBATCH --job-name=Test                  # Job name
#SBATCH --mem=16G                        # Job memory request. Different units can be specified using the suffix [K|M|G|T]
#SBATCH --time=4:00:00                   # Time limit 4 hours
#SBATCH --output=logs/stdout_%j.log      # Standard output and error log

set -x

source /gpfs/commons/groups/gursoy_lab/rweiner/ld/ldpip/bin/activate

python3 -V

CHROMOSOME=21
MINVALUE="1" # Value after 0.
DECIMALS=3

ldmat -l debug convert-chromosome "/gpfs/commons/groups/nygcfaculty/gursoy_knowles/UKB_ld/chr${CHROMOSOME}_*.npz" \
 /gpfs/commons/groups/gursoy_lab/rweiner/ld/data/processed/chr${CHROMOSOME}_m0_${MINVALUE}d${DECIMALS}.h5 \
 -m 0.${MINVALUE} -d $DECIMALS -c $CHROMOSOME

echo Finished!