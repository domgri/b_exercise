# Exercise

Build a simple project of data processing, model training and deploying to production.

### Repository structure:

- data_preparation.ipynb contains data inspection and manipulation
- BentoML contains files instructions on how to deploy model to Docker
- test_data_preparation.ipynb contains information how to format a sample query
- BentoML/prepared_data.csv contains analyzed and prepared data for model training
- test.json and test2.json query samples
- remaining files are for configuration

All instances can be inspected and run independently (BentoML contains prepared data to create a docker container).

### Requirements

- [x] Create a docker application where project will run => BentoML -> tutorial.ipynb
- [x] Create a deep learning model which would predict next month sales => LSTM model predicts revenue for the next 30 days
- [x] Create endpoint for November information to pass => test_data_preparation.ipynb
- [x] Structurize project and write clean code
- [ ] Optimise model for the fastest response
- [ ] Compare endpoint speed with other deployment frameworks (e.g. Fast API)
