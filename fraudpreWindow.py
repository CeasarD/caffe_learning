from tkinter import *
import seaborn as sns
import matplotlib.pyplot as plt
import method
from tkinter.scrolledtext import *
import fraud_predict as fp
class myApp():
    def __init__(self):
        pass


    def extract(self,addr,rule_str):
        pass

    def save(self,path):
        pass

    def testdraw(self,_data):
        sns.set(style='ticks')
        #exercise=sns.load_dataset('exercise')
        g=sns.factorplot(x=_data.columns[0],y=_data.columns[1],data=_data)#将frame数据用于绘图
        plt.show()
    def mymain(self):
        root=Tk()
        root.title('分析器')
        root.geometry("%dx%d+%d+%d"%(700,500,200,200))
        rdb=Button(root,text='read data',command=
                   lambda:self.testdraw(method.readFile(
                       'C:\\Users\\Ceasar\\Downloads\\UNRATE.csv')),width=40)#用lambda来给事件传参
        rdb.pack()
        textPad =ScrolledText(root, width=250, height=240)
        textPad.insert(constants.END, chars = str(method.readFile(
            'C:\\Users\\Ceasar\\Downloads\\UNRATE.csv')))
        textPad.pack()
        mainloop()



if __name__=='__main__':
    myapp=myApp()
    myapp.mymain()
