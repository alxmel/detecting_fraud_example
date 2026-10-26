# Data

The dataset used in this project is **intentionally not included** in this repository. Everything in this folder except this README is excluded by `.gitignore`.

## Why the data isn't here

- **Synthetic, but not redistributed.** The project was built with synthetic data. Even so, I chose not to publish the dataset.
- **Column names replaced.** All column names were changed from their original names. The names used in this repository, including those in the table below, are stand-ins.
- **Notebook outputs removed.** For the same reason, all cell outputs in `anomaly_detection.ipynb` were cleared before publishing.

## Expected input format

To run the notebook on your own data, supply one **pipe-delimited (`|`) CSV file** and point `DATA_PATH` (in the notebook's first code cell) at it. Each row of the file must be one **claim line item**, meaning one service billed on a claim.

| Column | Description |
|---|---|
| `CLAIM_ID` | Claim identifier, repeated on every line item of the same claim |
| `PATIENT_ID` | Patient identifier |
| `DOCTOR_ID` | Doctor identifier |
| `PRACTICE_ID` | Identifier of the doctor's practice |
| `CLAIM_DOLLAR_AMOUNT_PAID` | What was paid for the claim |
| `CLAIM_SERVICES_FULL_SUBMISSION` | What the doctor billed |
| `LINE_NUM` | Line item number |
| `LINE_SERVICE_COUNT` | Units of service billed on the line |
| `PRIMARY_EXAMINATION_CODE` | Main examination code |
| `EXAMINATION_CODE1` … `EXAMINATION_CODE12` | Full list of examination codes (up to 12) |

The claim-level columns (the IDs, the dollar amounts, and the examination codes) should be the same on every line of a claim. The notebook keeps the values from each claim's first line. It then rolls the line items up to one row per claim and derives `n_lines`, `n_services`, and `n_examinations` before training; see sections 2 and 3 of the notebook.
