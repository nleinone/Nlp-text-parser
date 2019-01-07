import tkinter
import subprocess

top = tkinter.Tk()

def RunScriptFile(n):
    if n == 1:
        subprocess.call("NaiveBayes_Classifier.py", shell=True)

    if n == 2:
        subprocess.call("SGD_Classifier.py", shell=True)

    if n == 3:
        subprocess.call("Knn_Classifier.py", shell=True)

    if n == 4:
        subprocess.call("NN_AA_Script.py", shell=True)

    if n == 5:
        subprocess.call("Preprocessing_Script.py", shell=True)



B1 = tkinter.Button(top, text ="Run NaiveBayes classifier", height = 2, width = 40, command = lambda: RunScriptFile(1))
B2 = tkinter.Button(top, text ="Stochastic gradient descent classifier", height = 2, width = 40, command = lambda: RunScriptFile(2))
B3 = tkinter.Button(top, text ="Knn classifier", height = 2, width = 40, command = lambda: RunScriptFile(3))
B4 = tkinter.Button(top, text ="PoS Analysis", height = 2, width = 40, command = lambda: RunScriptFile(4))
B5 = tkinter.Button(top, text ="Pre-processing dataset", height = 2, width = 40, command = lambda: RunScriptFile(5))

B1.pack()
B2.pack()
B3.pack()
B4.pack()
B5.pack()


top.mainloop()
