from tkinter import *
import seaborn as sns
import matplotlib.pyplot as plt
import method
from tkinter.scrolledtext import *
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
                   lambda:self.testdraw(method.readFile('C:\\Users\\Ceasar\\Downloads\\UNRATE.csv')),width=40)#用lambda来给事件传参
        rdb.pack()
##        m1=PanedWindow(root,showhandle=True,sashrelief=SUNKEN,sashwidth=5)
##        m1.pack(fill=BOTH,expand=1)
##        #输入与输出结果交互区m2
##            #输入交互区
##        m2=PanedWindow(orient=VERTICAL,showhandle=True,
##                       sashrelief=SUNKEN,sashwidth=5,width=400)
##        m1.add(m2)
##        #创建操作区
##        inputFrame=LabelFrame(m2,text='操作区',padx=5,pady=5,width=400,height=200)
##        #输入网址框
##        addr_label=Label(inputFrame,text='网址：')
##        addr_label.grid(row=0,column=1,sticky=E,padx=2,pady=2)
##        addr_input=Entry(inputFrame,width=40)
##        addr_input.grid(row=0,column=2,sticky=W,padx=2,pady=2)
##        #输入正则表达式框
##        rule_label=Label(inputFrame,text='正则表达式：')
##        rule_label.grid(row=1,column=1,sticky=E,padx=2,pady=2)
##        rule_input=Entry(inputFrame,width=40)
##        rule_input.grid(row=1,column=2,sticky=W,padx=2,pady=2)
##        #使用正则表达式提取链接按钮
##        extract_button=Button(inputFrame,text='提取',width=15)
##        extract_button.grid(row=2,column=2,sticky=E,padx=5)
##        #保存正确的链接结果按钮
##        save_button=Button(inputFrame,text='保存',width=15)
##        save_button.grid(row=2,column=2,sticky=W,padx=5)
##        #将操作区加入到m2界面中
##        m2.add(inputFrame)
##            #提取结果交互区
##        outputFrame=LabelFrame(m2,text='链接结果',padx=5,pady=5,width=400,height=300)
##        result_text=Text(outputFrame,width=54,height=23,bd=3)
##        result_text.grid(row=0)
##        #情况连接结果区按钮
##        clear_button=Button(outputFrame,text='清空链接结果区',
##                            command=(lambda x=result_text:x.delete(1.0,END)),width=40)
##        clear_button.grid(row=1,sticky=E,padx=5,pady=5)
##        m2.add(outputFrame)
##        #正则表达式规则区
##        rule_showFrame=LabelFrame(m1,text='正则表达式规则',padx=5,pady=5,width=290,height=190)
##        rules='. :表示匹配了换行符外的任何字符\n| :表示匹配正则表达式A或者B\n^ :1.(脱字符）匹配输入字符串的开始位置\n\
##2.如果设置了re.MULTILINE标志，^也匹配换行符之后的位置\n$ :1.匹配字符串输入的结束位置\n2.如果设置了re.MULTILINE标志，$也匹配换行符后面\
##的位置\n\\:1.将一个普通字符变成特殊字符，例如\\d表示匹配所有十进制数字\n2.解除元字符的特殊功能，例如\\.表示匹配点号本身\n\
##3.引用序号对应的子组所匹配的字符串\n[……] :字符类的表示方式，匹配所包含的任意一个字符\n注1.连字符-如果出现在字符串间表示字符范围的描述\
##；如果出现在首位则仅作为普通字符\n注2.特殊字符仅有反斜杠\\保持特殊含义，用于转义字符。其他特殊符号如*、+、？等均作为普通字符匹配\n\
##注3.脱字符^如果出现在首位则表示匹配不包含其中的任意字符；如果出现在字符中间就仅作为普通字符匹配\n{M,N} :M与N均非负整数，其中M<=N，表示\
##前面的RE表达式匹配M至N次\n1.{M,}表示至少匹配M次\n2.{,N}表示至少匹配N次\n3.{N}表示需要匹配N次\n*?,+?,??:默认情况下，*+？的匹配模式是\
##贪婪模式（会尽可能多的匹配符合规则的字符串）；而后面带有？的表示开启非贪婪模式。\n{M,N}?:表示开启非贪婪模式，只匹配M次。\n\
##(……):匹配圆括号中的正则表达式，或者表示一个子组\n注：子组的内容可以在匹配之后被\\数字再次引用\n举个例子：(\\w+)\\1可以匹配字符串“fish从\
## fishc.com”中的fishc fishc，\\1就表示前边（）中的子组一样的正则表达式\n(?……):（？开头的表示为正则表达式的扩展语法\n\
## (?:...):非捕获组，即该子组匹配的字符串无法从后边获取\n(?aiLmsux):1.在（？后可以紧跟a，i，L，m，s，u，x中的一个或者多个字符，只能在\
## 正则表达式使用\n2.每一个字符对应一种匹配标志：\nre-A表示只匹配ASCII字符\nre-I表示忽略大小写\nre-L表示区域设置\nre-M表示多行模式\nre-S\
## 表示匹配任意符号\nre-X:表示详细表达式\n只要包含上述这些字符就将会影响整个正则表达式的规则\n(?P<name>...):命名组，通过组的名字name就\
## 可以访问到子组匹配的字符串\n(?P=name):反向引用一个命名组，它匹配指定命名组匹配的任何内容\n(?#...)表示注释，括号内的内容将被忽略\n\
## (?=...):前向肯定断言，如果当前包含的正则表达式（这里以...表示)在当前位置匹配成功，则代表成功，否则失败，一旦该部分正则表达式被匹配\
## 引擎尝试过，就不会继续进行匹配了；剩下的模式在此断言开始的地方继续尝试\n举个例子：love（=dwb）只匹配love紧跟dwb的字符串\n'
##        msg_sb=Scrollbar(rule_showFrame)
##        msg_sb.pack(side=RIGHT,fill=Y)
##        msg_cav=Canvas(rule_showFrame,width=250,height=500,yscrollcommand=msg_sb.set)
##        msg_cav.pack(fill=BOTH)
##
        textPad =ScrolledText(root, width=250, height=240)
        textPad.insert(constants.END, chars = str(method.readFile('C:\\Users\\Ceasar\\Downloads\\UNRATE.csv')))
        textPad.pack() 
##        messageshow=Message(root,text=,width=230)
##        messageshow.pack()
##        msg_sb.config(command=msg_cav.yview)
##        m1.add(rule_showFrame)

        mainloop()



if __name__=='__main__':
    myapp=myApp()
    myapp.mymain()

