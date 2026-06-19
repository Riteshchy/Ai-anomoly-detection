import tkinter as tk
from gui import CyberSecApp

if __name__ == "__main__":
    root = tk.Tk()
    # Optional: Set an icon if you have one
    # root.iconbitmap('icon.ico') 
    
    app = CyberSecApp(root)
    root.mainloop()