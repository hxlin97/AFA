![image](https://user-images.githubusercontent.com/58459755/192413071-38379ad0-3117-4b70-9720-61e8dd30c3a4.png)

1. System Requirements:

System: ubuntu version: 18.04
Software: python version: 3.7.11
We run these codes using Nvidia Titan Xp
We attached a small dataset in dataset/regression/sample.

2. Installation guide
One can setup the environment using anaconda, the typical install time is less than an hour, but it depends on the internet speed.
the command is:
conda env create -f environment.yml
then activate the environment through command:
conda activate tfq

3. Demo
One may run our code using the script train.py inside the folder main.
An example command is: 
python train.py regression sample 0 10 6 6 32 32 1e-4 0.99 10 2000 test_log
corresponds to regression on dataset "sample", 
with 0 modification, 
10 bond dimension, 
the hyperparameter for encoding MPS and MPO are 6, 
the batch size is 32 for test and train
the learning rate is 1e-4
the decay interval is 10
the number of epoches is 2000
the output is saved in test_log.txt in the folder output.

Notice that the output entirely depends on the TN contraction results, which comes from the matrix production. If the TN contraction results for each parts are known, they can be directly added without abundant calculations.
Like the molecule C formed by A and B 
TN_contraction results  of C can be directly obtained by the adding of TN_contraction results of A and TN_contraction results of B.

The expected output is:
"""
Epoch	Time(sec)	Loss_train	MAE_dev	MAE_test
1	24.78169422596693	51343368.59460449	57.13965818951954	59.022087139204984
2	48.64706811727956	5353620.6083984375	56.775621237657944	58.648000369387645
3	73.6783681162633	3326552.158023834	5.015739998483766	4.979550294293239
4	99.06186244916171	47647.09996366501	3.3494113207401592	3.2787066688676645
5	124.90278318338096	17699.269937992096	2.517828446209565	2.454029134666454
6	151.05004501715302	12096.532528162003	2.465032883207093	2.3985452272836016
7	176.67972720600665	9522.050415158272	1.651043267605267	1.6169817954329624
8	202.02122848993167	8056.3915021419525	1.707843640587938	1.673878884370058
9	227.56449614511803	7510.977852940559	2.220422622043446	2.2247274071723413
10	252.79444259498268	6773.867526173592	2.9346418660596587	2.892229346488506
11	278.38273243419826	5844.610919833183	1.7966580487804542	1.7408228666689212
12	304.6248670099303	5336.440129995346	1.3312141104153532	1.2990195098201773
13	330.0119587611407	4455.095820188522	1.1720185462831105	1.1518729470933586
14	355.9923691479489	4483.845223963261	1.1698516712231777	1.1505056550828765
15	381.89734259806573	3755.097065091133	0.9912931364761249	0.9648457645430164
16	408.04735829494894	3558.078502237797	1.0022028907965321	0.9856898039490972
17	433.52822756720707	2783.797519683838	2.1459914136417293	2.1208437060849024
18	459.1736560901627	2830.7541399002075	0.9078125189443205	0.9027856256210613
19	484.39730715192854	2497.115588068962	1.0574327796092151	1.0424868370341358
20	510.3121106121689	2146.0036296844482	2.012883636267944	2.017228299443516
"""
The expected run time is listed on the output.

4. Instructions for use
If you want to run AFA on your data:
    1. place the data in folders like dataset/regression/YOURSAMPLEDATA, rename them as data_test.txt and data_train.txt
    2. run thecodes using python train.py regression YOURSAMPLEDATA 0 10 6 6 32 32 1e-4 0.99 10 2000 test_log 
        Here "YOURSAMPLEDATA" shall be replaced by the name of your dataset.
