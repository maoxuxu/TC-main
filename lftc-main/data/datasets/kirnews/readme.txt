KirNews: Kirundi News Classification Dataset

Version 1, Updated 02/10/2020


LICENSE

The copyrigth of the news articles belongs to the orginal news sources.

DESCRIPTION

The KirNews dataset is collected from 8 news sources in total, 4 news websites and 4 newspapers from Burundi. It contains a total of 4,612 news articles which are distributed across 12 classes.
These classes are listed in classes.txt. It has a raw version and a preprocessed (cleaned) version which are both divided in 3,690 articles for the train set and 922 for the test set. 

The files train.csv and test.csv for the raw version contain samples as comma-sparated values. 
There are 6 columns in them, corresponding to class index (1 to 14), English labels (en_label), Kirundi labels (kir_label), url, title and content. 

The files train.csv and test.csv for the cleaned version also contain samples as comma-sparated values. 
However,there are 3 columns in them, corresponding to class index (1 to 14), title and content.