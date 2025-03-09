from tkinter import *
import time
import os  
from picamera2 import Picamera2
from tkinter import messagebox


# Initialize GUI
root = Tk()
root.title('Get Phenotype')
#root.iconbitmap('icon_images/Corn.ico')
root.geometry('620x220')  # 设置窗口大小

label_input = Entry(root,borderwidth=5, font=("Helvetica", 40))
label_input.grid(row=0, column=0)

# Initialize Picamera
picam2 = Picamera2()
video_config = picam2.create_video_configuration(buffer_count=6,
                                                 main={"size":(1072,1440)},
                                                 controls={"FrameDurationLimits":(25000,25000)})  #Control the fps to get specific frames

picam2.align_configuration(video_config)
picam2.configure(video_config)


# Define functions

def Video_Recording(label):
    
    picam2.start_and_record_video('dataset/' + label,duration=15, show_preview=True)
    picam2.stop_preview()

def Run():
    wd = os.path.join(os.getcwd(), 'dataset/')
    files = os.listdir(wd)
    
    ear_label = label_input.get() + ".mp4"
    
    if ear_label == ".mp4":
        messagebox.showwarning("WARNING!","文件名不能为空，请检查后重新输入！")
    elif ear_label in files:
        #label_input.delete(0,END)
        response = messagebox.askyesno("WARNING!","同名文件已存在，是否要覆盖？")
        if response == 1:
            return Video_Recording(ear_label)
        else:
            return
    else:
        #label_input.delete(0,END)
        return Video_Recording(ear_label)



# Make Buttons
button_Run = Button(root, text="Run",padx=20, command=Run, font=("Helvetica", 100))

button_Run.grid(row=1,column=0)

label_input.bind("<Return>", lambda event=None: Run())


root.mainloop()



