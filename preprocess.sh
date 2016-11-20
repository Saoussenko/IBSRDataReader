#!/usr/bin/env bash
source /etc/fsl/5.0/fsl.sh
FILES="/home/siavash/programming/thesis/nifti_files/*"
PREPROCESSED_FILES_ADDRESS="/home/siavash/programming/thesis/preprocessed_nifti_files/"

for f in $FILES
do
  if [[ $f != *'lbl'* ]];
  then 
    echo "Pre-processing $f"
    PREPROCESSED_FILE=$(basename $f .nii)
    f2="${PREPROCESSED_FILES_ADDRESS}${PREPROCESSED_FILE}_preprocessed.nii"
    $(bet ${f} ${f2})
    $(susan ${f2} -1 3 3 1 0 ${f2})
  fi
done
