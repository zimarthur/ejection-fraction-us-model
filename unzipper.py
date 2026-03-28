import os
import gzip
import shutil

# caminhos base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_OFICIAL = os.path.join(BASE_DIR, "..", "dataset_oficial")
DATASET_OUTPUT = os.path.join(BASE_DIR, "..", "dataset")

# arquivos que queremos extrair
FILES_TO_EXTRACT = [
    "2CH_half_sequence.nii.gz",
    "2CH_half_sequence_gt.nii.gz",
    "4CH_half_sequence.nii.gz",
    "4CH_half_sequence_gt.nii.gz",
]

# cria pasta dataset caso não exista
os.makedirs(DATASET_OUTPUT, exist_ok=True)

# percorre pacientes
for patient in sorted(os.listdir(DATASET_OFICIAL)):

    patient_path = os.path.join(DATASET_OFICIAL, patient)

    if not os.path.isdir(patient_path):
        continue

    print(f"Processing {patient}")

    # cria pasta destino do paciente
    output_patient_path = os.path.join(DATASET_OUTPUT)
    os.makedirs(output_patient_path, exist_ok=True)

    for suffix in FILES_TO_EXTRACT:

        filename = f"{patient}_{suffix}"
        input_file = os.path.join(patient_path, filename)

        if not os.path.exists(input_file):
            print(f"  Missing: {filename}")
            continue

        output_filename = filename.replace(".gz", "")
        output_file = os.path.join(output_patient_path, output_filename)

        # descompacta
        with gzip.open(input_file, "rb") as f_in:
            with open(output_file, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        print(f"  Extracted: {output_filename}")

print("Done.")