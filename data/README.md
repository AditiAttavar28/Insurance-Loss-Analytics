# Data

Two files of auto-insurance policy records, provided by the course instructors for the DSO 530 group project (USC Marshall School of Business, May 2026). The project materials do not document the original source of the data or a license for it.

| File | Rows | Columns | Contents |
|---|---|---|---|
| `insurance_train2026.csv` | 39,928 | 28 | Policies with claim information; used to build the targets and train the models |
| `insurance_test2026.csv` | 13,310 | 23 | Policies without claim information; used for the final predictions |

Column names are anonymized as `X.1`–`X.28`. The test file does not contain `X.1` (policy ID) or `X.15`–`X.18` (the claim columns).

## Target variables

The three prediction targets are computed from the claim columns in the training file:

| Target | Meaning | Formula |
|---|---|---|
| LC — loss cost | Average cost per claim; 0 if the policy had no claims | `X.15 / X.16` if `X.16 > 0`, else 0 |
| HALC — historically adjusted loss cost | LC scaled by the policy's historical claim-rate factor | `LC × X.18` |
| CS — claim status | Whether the policy had at least one claim | 1 if `X.16 > 0`, else 0 |

In the training data, 11.23% of policies had a claim. LC has a mean of 64.56 and a maximum of 35,191.03; HALC has a mean of 117.26 and a maximum of 78,131.46.

## Data dictionary

Descriptions come from the labels and comments used in the project notebook. Columns marked "not documented" are not described anywhere in the project materials, and their meaning is not assumed here.

| Column | Description | Notes |
|---|---|---|
| `X.1` | Policy ID | Training file only; not used as a feature |
| `X.2` | Policy start date | Used to derive policy duration |
| `X.3` | Date field — not documented | Not used |
| `X.4` | Next renewal date | Used to derive policy duration |
| `X.5` | Driver date of birth | Used to derive driver age |
| `X.6` | Driving license issue date | Used to derive license years |
| `X.7` | Sales channel | Coded 0/1 |
| `X.8` | Years with insurer | |
| `X.9` | Number of policies | |
| `X.10` | Maximum policies ever held | |
| `X.11` | Maximum products held | |
| `X.12` | Policies cancelled | |
| `X.13` | Payment method | Coded 0/1 |
| `X.14` | Net premium | |
| `X.15` | Total claim cost | Training file only; used to compute LC |
| `X.16` | Number of claims | Training file only; used to compute LC and CS |
| `X.17` | Not documented | Training file only; not used |
| `X.18` | Historical claim-rate factor | Training file only; used to compute HALC |
| `X.19` | Vehicle type | 1 = motorbike, 2 = van, 3 = passenger car, 4 = agricultural vehicle |
| `X.20` | Urban / rural indicator | Coded 0/1 |
| `X.21` | Multiple drivers indicator | Coded 0/1 |
| `X.22` | Vehicle registration year | |
| `X.23` | Horsepower | |
| `X.24` | Cylinder capacity | |
| `X.25` | Vehicle value | |
| `X.26` | Number of doors | |
| `X.27` | Fuel type | P = petrol, D = diesel |
| `X.28` | Vehicle weight | |

The meaning of the 0/1 codes for `X.7`, `X.13`, `X.20` and `X.21` is not documented. Dates are stored as `DD/MM/YYYY`.

## Cleaning applied in the notebook

- **Training file:** 453 rows have no fuel type (`X.27`). In all of them horsepower (`X.23`) and number of doors (`X.26`) are also 0. These rows were dropped, leaving 39,475 policies.
- **Test file:** 140 rows have no fuel type. They could not be dropped, because every test policy needs a prediction, so the missing values were filled with a Random Forest classifier trained on the vehicle columns (130 predicted petrol, 10 diesel).
- There are no duplicate rows in either file and no other missing values.

## Using the files

The notebook reads both files from this folder with relative paths (`../data/`). The notebook is a record of the original analysis and does not run from top to bottom, because some of the original code is missing; see the Limitations section of the main [README](../README.md).
