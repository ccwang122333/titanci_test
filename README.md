# titanci_test
kaggle_practice
处理过程的特点是：对age的缺失采用RF回归拟合；cabin按编号首字母分等级，缺失采用RF分类拟合
1.data_pre 数据预处理
2.datapre_func 数据预处理函数，处理test函数
3.feature_select 特征重要性分析，使用RS
4.main 处理test.csv(RF算法
5.意义：练习对data的前期处理，主要是特征哑处理、缺失值补齐，以及对经典分类算法的熟悉；
6.不足：未考虑特征的实际意义，特征选择较随意，对算法的结果不能进行优化
