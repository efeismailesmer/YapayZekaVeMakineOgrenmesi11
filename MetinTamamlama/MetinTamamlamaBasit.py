import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QTextEdit, QLineEdit, QPushButton, QVBoxLayout, QWidget
from transformers import pipeline
import time
from threading import Thread
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QTextCursor

class ChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # UI bileşenlerini oluştur
        self.setWindowTitle('Sohbet Penceresi')
        self.setGeometry(100, 100, 400, 500)

        # Text edit (sohbet ekranı)
        self.chat_display = QTextEdit(self)
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("""
            background-color: #2E2E2E;
            color: white;
            font-size: 14px;
            padding: 10px;
            border: 1px solid #333;
        """)

        # User input (kullanıcı metni girişi)
        self.user_input = QLineEdit(self)
        self.user_input.setStyleSheet("font-size: 14px; padding: 5px;")

        # Mesaj gönderme butonu
        self.send_button = QPushButton('Gönder', self)
        self.send_button.setStyleSheet("font-size: 14px; background-color: #4CAF50; color: white;")

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.chat_display)
        layout.addWidget(self.user_input)
        layout.addWidget(self.send_button)

        # Ana widget
        widget = QWidget(self)
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        # Modeli yükle
        self.text_generator = pipeline("text-generation", model="cenkersisman/gpt2-turkish-900m")

        # Butona tıklama olayı
        self.send_button.clicked.connect(self.generate_response)

        # Timer'ı başlat (boş olacak, yalnızca botun yazı ekleme zamanını yönetiyor)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_response)

        # Başka bir değişkenle bot yanıtını yönet
        self.bot_response = ""
        self.current_char_index = 0
    def generate_response(self):
        # Kullanıcının girdiği metni al
        user_input = self.user_input.text()

        if user_input:
            # Kullanıcı mesajını sohbet ekranına ekle
            self.chat_display.append(f"<font color='green'>Sen: {user_input}</font>")
            
            # YZ yanıtını başlat
            self.chat_display.append("<font color='white'>Bot: </font>")  # Bot yazısına başlangıç

            # Modeli kullanarak metin tamamlama yap
            result = self.text_generator(user_input, max_length=150, num_return_sequences=1, truncation=True)
            self.bot_response = result[0]['generated_text']

            # Başlangıç için karakter indexini sıfırla
            self.current_char_index = 0

            # Timer'ı başlat
            self.timer.start(50)  # Her 50ms'de bir karakter ekleyerek devam et

        # Giriş kutusunu temizle
        self.user_input.clear()

    def update_response(self):
        if self.current_char_index < len(self.bot_response):
            # Chat ekranına yeni bir karakter ekle
            self.chat_display.moveCursor(self.chat_display.textCursor().MoveOperation.End)  # Yazıyı sonuna taşı
            self.chat_display.insertPlainText(self.bot_response[self.current_char_index])

            # Bir sonraki karaktere geç
            self.current_char_index += 1
        else:
            # Yanıt tamamen yazıldığında timer'ı durdur
            self.timer.stop()

# Uygulama başlatma
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ChatWindow()
    window.show()
    sys.exit(app.exec())
