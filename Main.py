import tkinter as tk
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
from tkinter import messagebox
import os

# path='C:/Users/Dell/Deskroot/Tests/test1.jpg'
path=''
def ShowImage():
    fln= filedialog.askopenfilename(initialdir=os.getcwd(), title="Select Image",filetypes=(("JPG File","*.jpg"),("PNG File","*.png")))
    global path
    path=fln
    img = Image.open(fln)
    img.thumbnail((480,400))
    img = ImageTk.PhotoImage(img)
    lbl.configure(image= img)
    lbl.image= img

def detect():
    if path!='':
        # root.destroy()
        cmd = "detect.py --weights ./data/custom-yolov4-detector_final.weights --framework tf --size 416 --image " + path
        os.system(cmd)
    else:
        messagebox.showwarning("Warrning", "Please Select Image First")

root = Tk()

frm = Frame(root)
frm.pack(side= BOTTOM , padx=15,pady=15)

lbl=Label(root)
lbl.pack()

btn=Button(frm, text="Browse Smear Image",command=ShowImage)
btn.pack(side= tk.LEFT)

btn2= Button(frm,text="Detect", command=detect)
btn2.pack(side=tk.LEFT,padx=10)

root.title("Data Entry")
root.geometry("500x500")
root.mainloop()


