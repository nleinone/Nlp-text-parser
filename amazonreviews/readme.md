# comment utiliser #
Copy Amazon data set under in this folder (https://www.kaggle.com/bittlingmayer/amazonreviews/home)

Dataset is expected to be in folders train/train.ft.txt and test/test.ft.txt

## For 25000 positive and 25000 negative reviews ##
py .\file_sorter.py -i train -s 25000 

py .\file_sorter.py -i test -s 25000


## For whole dataset ##
py .\file_sorter.py -i test

py .\file_sorter.py -i train

