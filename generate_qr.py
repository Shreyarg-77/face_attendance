import qrcode

# Replace with your actual Railway public URL (e.g., https://your-app-name.up.railway.app/)
app_url = 'https://faceattendancesystem-production.up.railway.app/'

qr = qrcode.QRCode(version=1, box_size=10, border=5)
qr.add_data(app_url)
qr.make(fit=True)
img = qr.make_image(fill='black', back_color='white')
img.save('attendance_qr.png')  # Saves the QR code as a PNG file
print("QR code saved as 'attendance_qr.png'")