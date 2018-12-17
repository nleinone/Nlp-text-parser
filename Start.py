import tkinter
import subprocess

top = tkinter.Tk()

def RunScriptFile():
   subprocess.call("Text_parser_25_11.py", shell=True)

B1 = tkinter.Button(top, text ="Run NaiveBayes classifier", height = 2, width = 40, command = RunScriptFile)
B2 = tkinter.Button(top, text ="Stochastic gradient descent classifier", height = 2, width = 40, command = RunScriptFile)
B3 = tkinter.Button(top, text ="Knn classifier", height = 2, width = 40, command = RunScriptFile)

B1.pack()
B2.pack()
B3.pack()

top.mainloop()
