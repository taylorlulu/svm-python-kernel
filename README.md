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

Some results are shown below:\n
![Linear dual space](https://user-images.githubusercontent.com/49311079/195758898-8ff8b3a7-8559-4e3a-9e7c-07e78cbb1655.png)
![Linear dual space曲线分割](https://user-images.githubusercontent.com/49311079/195758910-cb1dd085-5128-49af-9dba-2bafcdddaf94.png)
![Linear dual space直线分割](https://user-images.githubusercontent.com/49311079/195758916-29c60bdc-72ba-4822-8bb8-f7605df91e96.png)
![Nonlinear soft margin C=100 00,kernel=polynomial曲线分割](https://user-images.githubusercontent.com/49311079/195758947-082bba31-087e-4c58-a580-949a8d6bfb66.png)
![Nonlinear soft margin C=100 00,kernel=poly曲线分割](https://user-images.githubusercontent.com/49311079/195759012-dcc192d7-ed68-4018-94a4-dd0f3cc5889a.png)
