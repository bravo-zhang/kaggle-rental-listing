# kaggle-rental-listing
Two Sigma Connect: Rental Listing Inquiries -- How much interest will a new rental listing on RentHop receive?

## How to use
1. Put `train.json` and `test.json` under `/input`.
2. `python feature_eng.py` will generate pre-processed data in 2 csv files in `/data`.
3. `python model/xgb.py` to run cross validation.  
`python model/xgb.py -s` to generate `submission.csv` in `/submission`.  
`python model/xgb.py -h` for more information.
