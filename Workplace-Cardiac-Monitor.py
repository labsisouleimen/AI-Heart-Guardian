import customtkinter as ctk
from PIL import Image, ImageTk
from tkinter import filedialog
import tkinter as tk
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Flatten, Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import wfdb 
from tkinter import messagebox
# 📌 تحميل StandardScaler والنموذج المدرب
scaler = joblib.load("scaler.pkl")
model = tf.keras.models.load_model("ecg_cnn_lstm_model2.h5")
# إنشاء نافذة CustomTkinter
apphealth = ctk.CTk(fg_color="#ffffff")
apphealth.title("WorkerHealth")
apphealth.iconbitmap("mecg/imgecg/favicon.ico")
apphealth.geometry("650x500")
apphealth.resizable(False, False)

def center_window(window):
    window.update_idletasks()
    width = window.winfo_width()
    height = window.winfo_height()
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = (screen_width // 2) - (width // 2)
    y = (screen_height // 2) - (height // 2)
    window.geometry(f'{width}x{height}+{x}+{y}')

center_window(apphealth)

def on_main_window_close():
    apphealth.destroy()

apphealth.protocol("WM_DELETE_WINDOW", on_main_window_close)

topbar = ctk.CTkFrame(apphealth, height=250, corner_radius=0, fg_color="#ffffff")
topbar.pack(side="bottom", fill="x")

def list_button_topbar(parent, icon_path, text, command=None):
    icon = ctk.CTkImage(light_image=Image.open(icon_path), size=(35, 35))
    btn = ctk.CTkButton(parent, image=icon, text=text, width=145, height=100,
                        fg_color="#ffffff", hover_color="#ffffff", compound="top",
                        anchor="center", corner_radius=5, font=("", 14), command=command,text_color="black")
    btn.pack(side="left", padx=35, pady=30)

    def on_enter(e):
        btn.configure(fg_color="#1f1f1f", text_color="white")

    def on_leave(e):
        btn.configure(fg_color="#ffffff", text_color="black")

    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)

    return btn

def on_menu_choice(choice):
    print(f"Selected: {choice}")

def show_menu(event):
    menu.tk_popup(event.x_root, event.y_root)




def open_new_window():
    new_win = ctk.CTkToplevel()
    new_win.geometry("300x200")
    new_win.title("New Window")
    label = ctk.CTkLabel(new_win, text="This is a new CTkToplevel window")
    label.pack(pady=20)
# إنشاء قائمة منبثقة Menu من tkinter
menu = tk.Menu(apphealth, tearoff=0)
menu.add_command(label="Setting", command=open_new_window)
menu.add_separator()
menu.add_command(label="Exit", command=apphealth.destroy)
menu.add_command(label="About Us")

icon5 = ctk.CTkImage(light_image=Image.open("mecg/imgecg/menu.png"), size=(30, 30))
#btn5 = ctk.CTkButton(apphealth, text="Menu", width=100, height=50)
btn5=ctk.CTkButton(apphealth, image=icon5, text="Menu", width=145, height=100,
                        fg_color="#ffffff", hover_color="#ffffff", compound="left",
                        anchor="center", corner_radius=5, font=("", 14), text_color="black")
btn5.place(x=500, y=10)  # وضع الزر في موقع معين
# عند ضغط الزر، تظهر القائمة المنبثقة
btn5.bind("<Button-1>", show_menu)



# 📌 دالة لفتح ملف .dat ومعالجته
def open_file():
    filepath = filedialog.askopenfilename(
        title="Select ECG File (no extension)",
        filetypes=(("All files", "*.*"),)
    )
    if filepath:
        # 📌 استخراج اسم الملف فقط بدون الامتداد والمسار
        import os
        base = os.path.basename(filepath)   # مثل: 19830
        file_name = os.path.splitext(base)[0]
        directory = os.path.dirname(filepath)

        print(f"📁 Selected: {file_name} from {directory}")

        try:
            record = wfdb.rdrecord(file_name)  # يقرأ الملف من المسار
            signal = record.p_signal[:, 0]

            if len(signal) < 1800:
                padded_signal = np.zeros(1800)
                padded_signal[:len(signal)] = signal
                signal = padded_signal
            else:
                signal = signal[:1800]

            signal_scaled = scaler.transform(signal.reshape(1, -1))
            signal_scaled = signal_scaled.reshape(1, 1800, 1)

            prediction = model.predict(signal_scaled)
            predicted_class = np.argmax(prediction)

            if predicted_class == 0:
                #print("✅ الإشارة طبيعية")
                messagebox.showinfo("Analysis Result", "✅ The ECG signal is normal.")
            else:
                #print("❌ الإشارة غير طبيعية")
                messagebox.showwarning("Analysis Result", "❌ The ECG signal is abnormal.")

        except Exception as e:
            #print("⚠️ خطأ أثناء قراءة الملف:", e)
            messagebox.showerror("Error", f"⚠️ An error occurred while reading the file:\n{e}")

def home():
    messagebox.showinfo("Hello","Welcome to the WorkerHealth platform! We're here to help you easily monitor and analyze your ECG signals")

def contactus():
    rot = ctk.CTkToplevel()
    rot.title("Contact Us")
    rot.geometry('600x500')
    rot.resizable(False, False)
    rot.transient()
    rot.configure(fg_color="white")
    center_window(rot)
    content_frame = ctk.CTkFrame(rot, fg_color="transparent")
    content_frame.pack(padx=10, pady=10, fill="both", expand=True)
    text_help = "If you need help call for Us"
    text_label1 = ctk.CTkLabel(content_frame, text=text_help, font=ctk.CTkFont(size=13), wraplength=450, justify="left",text_color="black")
    text_label1.pack(padx=10, pady=10, anchor="w")
    contacts = [
        "BOURBIA Oussila Ing/Pro/Doc: 0699 18 72 19",
        "LABSI Mohamed Souliemen Ing/Dev/Pro: 0797 27 02 47",
        "LABSI Mehdi Ing: 0549 16 08 82",
        "Berkane Mohammed Ing: 0796 23 61 50",
        "KHALFI Kaouther Anfel Ing: 0792 82 51 55 ",
        "HABCHI Boutheina Nesrine Ing: 0796 75 41 12"
    ]
    for contact in contacts:
        ctk.CTkLabel(content_frame, text=contact, text_color="black", anchor="w").pack(anchor="w", padx=20, pady=5)


# إضافة الأزرار
list_button_topbar(topbar, "mecg/imgecg/home.png", "Home", command=home)
list_button_topbar(topbar, "mecg/imgecg/pin-location.png", "Contact Us",command=contactus)
list_button_topbar(topbar, "mecg/imgecg/monitor.png", "Healthy", command=open_file)

# شعار التطبيق
img_apphealth = Image.open("mecg/imgecg/WorkerHealth Logo Design.png")
health_image = ctk.CTkImage(dark_image=img_apphealth, size=(250, 250))
healthimg_label = ctk.CTkLabel(apphealth, image=health_image, text="")
healthimg_label.pack(pady=25, anchor="center")

apphealth.mainloop()

