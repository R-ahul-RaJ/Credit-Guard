Download VS CODE : https://code.visualstudio.com/docs/?dv=win64user

STEPS TO DEPLOY IN LOCAL:

OPEN Terminal ctrl + shift + ~

Always choose command prompt and delete powershell
1. conda create -n test python=3.10 -y


2. conda activate test

3. create a requirements.txt file

4. add required packages in requirements.txt file

5. pip install -r requirements.txt

6. create a predict.py file

7. prepare the codes in predict.py file

8. streamlit run predict.py