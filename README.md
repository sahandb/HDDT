# HDDT Hellinger Distance Decision Tree
Implementation of Hellinger Distance Decision Tree(HDDT) with some COVID-19 datasets


HDDT is a decision tree based on Hellinger distance, and it is suitable for unbalanced data.
Read the attached documents for more information about the HDDT algorithm and its implementation.

We evaluate the performance of HDDT on the Coronavirus data set.

Since the data set is unbalanced, we should use Precision, Recall, F-measure, and AUC measures to evaluate the performance of our tree. It is worth mentioning that these measures are one-class measures; in other words, we should compute these metrics just for the minority class.

We split the data set to train and test parts. Use 70% of the data for training phase and the remaining 30% for testing phase. Run our codes for 10 individual runs and report the average of 10 runs for each performance metric.

Note that the original version of the data set (Con_Covid-19.csv) has continuous features, which should be used to train these four classifiers(Naïve Bayes, One-nearest-neighbor (OneNN), linear SVM, and kernel SVM (use RBF kernel)) for compare the performance of the trees with them. We use the discretized version of the data (Dis_Covid-19.csv) to train the HDDTs.

# steps:

At first I delete 11 feature from the dataset because the algorithm converged so fast, and then read the data and chunk data into 70percent for train and 30 percent for test

Then we turnd the data every time for train to 2 class because hddt is binary classifier and our data has 3 class and 

After thet I create 2 function for run one of them is one vs one and the other is one vs rest

And then I choose 3 cut off by majority and the cutoffs are 10 50 and 500 and use 4 max height for the problem (2 3 4 5)

After that I use 4 classifires : naïve bayes – ONN – SVM kernel (rbf) – svc

And at the end I run my code 10 times for every steps and get resault
I commented the description you probably in my code too.

# HDDT Hellinger distance decision tree
Binary classifier 
And it good for imbalanced data
Because it use Hellinger distance as measurement for find the best root for tree

What are the properties of the HDDT algorithm? Why is it suitable for unbalanced data?
Don’t ignore minority class and imbalance data and get the best performance on these type of data and we don’t have problem with splitting criterion in this tree because that takes a different splitting criterion that has proven to be skew insensitive.

What are the differences between Hellinger distances, Gini index, and information gain? 
These two measurements are heavily skew sensitive1and have a large disadvantage in terms of classification performance when operating on minority groups  and in hddt that takes a different splitting criterion into account has proven to be skew insensitive.

Is pruning lead to better results? Why?
trees were pruned, leaves with few observations would be eliminated and are most likely the ones associated with the minority class. Hence, an essential part when working with unbalanced data is to maintain the tree as a whole. Because unpruned trees are capable of finding more splits in the dataset and further differentiate the class of interest, i.e. an algorithm has a greater chance of discovering more unusual split.


