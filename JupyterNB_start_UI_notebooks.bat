:: enable ML_TOP96_2 environment
set root=C:\workspace\100_Website\100_Ipynb_plus_mybinder\ipynb_mybinder\Scripts
call %root%\activate.bat %root%

:: start jupyter
call cd "C:\workspace\100_Website\100_Ipynb_plus_mybinder"
call jupyter notebook

pause