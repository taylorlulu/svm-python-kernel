# svm-python-kernel
Including hard margin svm(original space and dual space), soft margin svm. The problems were solver by cvxopt library and sklearn respectively.
The main target of this project is to realize road segmentation through svm.
Several ways were provided in the svmpy folder.
The svm_original_space.py is to solve hard margin svm in original space.
The svm_original_space.py is to solve hard margin svm in dual space.
The svm_soft_margin.py is a soft margin svm.
The svm_soft_margin.py introduce kernel function on the basis of svm_soft_margin.py.
You can test it through the svm_qp_test.jupyter or svm_sklearn_test.jupyter files.
If you use the svm_qp and svm_sklearn code, please input data through mouse, right click on the non-road area and left click on the road area.
The program will run on their own.
It should be noted that the svm_original_space may have some problems to solve linearly inseparable problems.
