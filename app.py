from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import face_recognition
import cv2
import numpy as np
from deepface import DeepFace
import numpy as np
import pickle
import os
from datetime import datetime, timedelta
import json
import csv
from io import StringIO
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import time
import qrcode
from io import BytesIO
import logging

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a random string
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///models/attendance.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

logging.basicConfig(level=logging.INFO)
app.logger.info("App starting...")

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Ensure models folder exists
if not os.path.exists('models'):
    os.makedirs('models')

# Global variables for face recognition
known_face_encodings = []
known_face_student_ids = []
models_loaded = False  # Flag for lazy model loading

# User class for Flask-Login
class Admin(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(200))
    class_name = db.Column(db.String(100))
    email = db.Column(db.String(100))

    def __init__(self, id, username, password, class_name, email=None):
        self.id = id
        self.username = username
        self.password = password
        self.class_name = class_name
        self.email = email

class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    class_name = db.Column(db.String(100))
    enrolled_by_admin_id = db.Column(db.Integer, db.ForeignKey('admin.id'))
    encoding = db.Column(db.Text)
    class_display_id = db.Column(db.Integer)

class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'))
    date = db.Column(db.String(20))
    time = db.Column(db.String(20))

@login_manager.user_loader
def load_user(user_id):
    return Admin.query.get(int(user_id))

# Load known faces (encodings only, lightweight)
def load_known_faces():
    global known_face_encodings, known_face_student_ids
    known_face_encodings = []
    known_face_student_ids = []
    students = Student.query.filter(Student.encoding.isnot(None)).all()
    for student in students:
        encoding = np.array([float(x) for x in student.encoding.split(',')])
        known_face_encodings.append(encoding)
        known_face_student_ids.append(student.id)
    app.logger.info(f"Loaded {len(known_face_encodings)} face encodings.")

# Lazy load DeepFace models (on-demand)
def load_models_if_needed():
    global models_loaded
    if not models_loaded:
        try:
            DeepFace.build_model('VGG-Face')  # Force model build
            models_loaded = True
            app.logger.info("DeepFace models loaded on-demand.")
        except Exception as e:
            app.logger.error(f"Model loading error: {e}")

# Create tables and load encodings on startup
time.sleep(10)  # Delay for DB stability
with app.app_context():
    try:
        db.create_all()
        load_known_faces()
        app.logger.info("App initialized successfully.")
    except Exception as e:
        app.logger.error(f"Startup error: {e}")

# Routes
@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/register_admin', methods=['GET', 'POST'])
def register_admin():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])
        class_name = request.form['class_name']
        email = request.form['email']
        new_id = db.session.query(db.func.max(Admin.id)).scalar() or 0 + 1
        new_admin = Admin(id=new_id, username=username, password=password, class_name=class_name, email=email)
        try:
            db.session.add(new_admin)
            db.session.commit()
            flash('Admin registered successfully!')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash(f'Registration error: {e}')
    return render_template('register_admin.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = Admin.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Invalid credentials.')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('welcome'))

@app.route('/dashboard')
@login_required
def dashboard():
    status_file = 'models/kiosk_active.txt'
    kiosk_active = os.path.exists(status_file)
    kiosk_admin = ''
    if kiosk_active:
        try:
            with open(status_file, 'r') as f:
                kiosk_admin = f.read()
        except Exception as e:
            app.logger.error(f"Error reading kiosk status: {e}")
    return render_template('dashboard.html', kiosk_active=kiosk_active, kiosk_admin=kiosk_admin)

@app.route('/attendance')
@login_required
def attendance():
    from datetime import date
    today = date.today().strftime('%Y-%m-%d')
    
    view = request.args.get('view', 'today')
    selected_date = request.args.get('date', today if view == 'today' else '')
    name_query = request.args.get('name', '').strip()
    date_query = request.args.get('date', today if view == 'today' else '').strip()
    
    if view == 'history':
        date_records = db.session.query(Attendance.date, db.func.count(Attendance.id)).join(Student).filter(Student.class_name == current_user.class_name).group_by(Attendance.date).order_by(Attendance.date.desc()).all()
        return render_template('attendance.html', view=view, date_records=date_records)
    elif view == 'date' and selected_date:
        query = Attendance.query.join(Student).filter(Student.class_name == current_user.class_name, Attendance.date == selected_date)
        if name_query:
            query = query.filter(Student.name.like(f'%{name_query}%'))
        records = query.order_by(Attendance.time.desc()).all()
        return render_template('attendance.html', view=view, records=records, selected_date=selected_date)
    else:
        query = Attendance.query.join(Student).filter(Student.class_name == current_user.class_name, Attendance.date == date_query)
        if name_query:
            query = query.filter(Student.name.like(f'%{name_query}%'))
        records = query.order_by(Attendance.time.desc()).all()
        return render_template('attendance.html', view='today', records=records, selected_date=date_query)

@app.route('/students', methods=['GET', 'POST'])
@login_required
def students():
    students_list = Student.query.filter_by(class_name=current_user.class_name).order_by(Student.class_display_id).all()
    enrolled_status = {student.id: (student.id in known_face_student_ids) for student in students_list}
    
    if request.method == 'POST':
        name = request.form['name']
        max_display_id = db.session.query(db.func.max(Student.class_display_id)).filter(Student.class_name == current_user.class_name).scalar() or 0
        class_display_id = max_display_id + 1
        new_id = db.session.query(db.func.max(Student.id)).scalar() or 0 + 1
        new_student = Student(id=new_id, name=name, class_name=current_user.class_name, enrolled_by_admin_id=current_user.id, class_display_id=class_display_id)
        db.session.add(new_student)
        db.session.commit()
        flash('Student added!')
    
    return render_template('students.html', students=students_list, enrolled_status=enrolled_status)

@app.route('/delete_student/<int:student_id>', methods=['POST'])
@login_required
def delete_student(student_id):
    student = Student.query.filter_by(id=student_id, class_name=current_user.class_name).first()
    if not student:
        flash('Student not found or not in your class.')
        return redirect(url_for('students'))
    
    Attendance.query.filter_by(student_id=student_id).delete()
    db.session.delete(student)
    db.session.commit()
    
    # Re-order class_display_id for remaining students
    remaining_students = Student.query.filter_by(class_name=current_user.class_name).order_by(Student.class_display_id).all()
    for i, student in enumerate(remaining_students, start=1):
        student.class_display_id = i
    db.session.commit()
    
    if student_id in known_face_student_ids:
        index = known_face_student_ids.index(student_id)
        known_face_encodings.pop(index)
        known_face_student_ids.pop(index)
    
    flash(f'Student {student.name} deleted successfully.')
    return redirect(url_for('students'))

@app.route('/edit_student/<int:student_id>', methods=['GET', 'POST'])
@login_required
def edit_student(student_id):
    student = Student.query.filter_by(id=student_id, class_name=current_user.class_name).first()
    if not student:
        flash('Student not found or not in your class.')
        return redirect(url_for('students'))
    
    if request.method == 'POST':
        name = request.form['name']
        class_name = request.form['class_name']
        if not name or not class_name:
            flash('All fields are required.')
            return render_template('edit_student.html', student={'id': student_id, 'name': student.name, 'class_name': student.class_name})
        
        student.name = name
        student.class_name = class_name
        db.session.commit()
        flash('Student updated successfully!')
        return redirect(url_for('students'))
    
    return render_template('edit_student.html', student={'id': student_id, 'name': student.name, 'class_name': student.class_name})

@app.route('/enroll_face/<int:student_id>', methods=['GET', 'POST'])
@login_required
def enroll_face(student_id):
    student = Student.query.filter_by(id=student_id, class_name=current_user.class_name).first()
    if not student:
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'status': 'error', 'message': 'Student not found or not in your class.'})
        flash('Student not found or not in your class.')
        return redirect(url_for('students'))
    
    if request.method == 'POST':
        image_data = request.form['image']
        try:
            import base64
            image = base64.b64decode(image_data.split(',')[1])
            np_arr = np.frombuffer(image, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            # Use face_recognition instead of DeepFace
            face_encodings = face_recognition.face_encodings(img)
            if face_encodings:
                face_encoding = face_encodings[0]
                encoding_str = ','.join(map(str, face_encoding))
                
                student.encoding = encoding_str
                db.session.commit()
                
                load_known_faces()
                
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({'status': 'success', 'message': f'Face enrolled for {student.name}!'})
                flash(f'Face enrolled for {student.name}!')
                return redirect(url_for('students'))
            else:
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return jsonify({'status': 'error', 'message': 'No face detected. Try again.'})
                flash('No face detected. Try again.')
        except Exception as e:
            app.logger.error(f"Error processing image: {e}")
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'status': 'error', 'message': 'Error processing image.'})
            flash('Error processing image.')
    return render_template('enroll_face.html', student=student.name)

@app.route('/insights')
@login_required
def insights():
    attendance_per_student = db.session.query(Student.name, db.func.count(Attendance.id)).join(Attendance).filter(Student.class_name == current_user.class_name).group_by(Student.id, Student.name).order_by(db.func.count(Attendance.id).desc()).all()
    
    totals = db.session.query(db.func.count(Attendance.id), db.func.count(db.distinct(Attendance.student_id))).join(Student).filter(Student.class_name == current_user.class_name).first()
    total_records = totals[0] if totals else 0
    unique_students = totals[1] if totals else 0
    
    insights_data = {
        'total_records': total_records,
        'unique_students': unique_students,
        'attendance_per_student': attendance_per_student
    }
    
    return render_template('insights.html', insights_data=insights_data, insights_json=json.dumps(insights_data))

@app.route('/blacklist')
@login_required
def blacklist():
    today = datetime.now()
    week_start = today - timedelta(days=today.weekday())
    week_end = week_start + timedelta(days=6)
    week_start_str = week_start.strftime('%Y-%m-%d')
    week_end_str = week_end.strftime('%Y-%m-%d')
    
    students_list = Student.query.filter_by(class_name=current_user.class_name).all()
    blacklisted_students = []
    for student in students_list:
        attendance_count = Attendance.query.filter(Attendance.student_id == student.id, Attendance.date.between(week_start_str, week_end_str)).count()
        percentage = (attendance_count / 5) * 100 if 5 > 0 else 0
        if percentage < 50:
            blacklisted_students.append((student.id, student.name, attendance_count, round(percentage, 1)))
    
    return render_template('blacklist.html', blacklisted_students=blacklisted_students, week_start=week_start_str, week_end=week_end_str)

@app.route('/kiosk_status', methods=['GET', 'POST', 'DELETE'])
@login_required
def kiosk_status():
    status_file = 'models/kiosk_active.txt'
    if request.method == 'POST':
        try:
            with open(status_file, 'w') as f:
                f.write(f"{current_user.username} ({current_user.class_name})")
            return jsonify({'status': 'active'})
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)})
    elif request.method == 'DELETE':
        try:
            if os.path.exists(status_file):
                os.remove(status_file)
            return jsonify({'status': 'inactive'})
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)})
    else:
        active = os.path.exists(status_file)
        admin = ''
        if active:
            try:
                with open(status_file, 'r') as f:
                    admin = f.read()
            except Exception as e:
                app.logger.error(f"Error reading kiosk status: {e}")
        return jsonify({'active': active, 'admin': admin})

@app.route('/export_attendance_csv')
@login_required
def export_attendance_csv():
    records = db.session.query(Attendance.id, Student.name, Student.class_name, Attendance.date, Attendance.time).join(Student).filter(Student.class_name == current_user.class_name).order_by(Attendance.date.desc(), Attendance.time.desc()).all()
    
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow([f'Attendance Report for {current_user.class_name}'])
    writer.writerow(['ID', 'Student Name', 'Class Name', 'Date', 'Time'])
    for record in records:
        writer.writerow([record[0], record[1], record[2], f'"{record[3]}"', record[4]])
    
    output.seek(0)
    return Response(output.getvalue(), mimetype='text/csv', headers={'Content-Disposition': 'attachment; filename=attendance.csv'})


@app.route('/send_attendance_mail', methods=['POST'])
@login_required
def send_attendance_mail():
    recipient_email = current_user.email
    if not recipient_email:
        return jsonify({'message': 'No email found for this admin. Please update your profile.'})
    
    records = db.session.query(Attendance.id, Student.name, Student.class_name, Attendance.date, Attendance.time).join(Student).filter(Student.class_name == current_user.class_name).order_by(Attendance.date.desc(), Attendance.time.desc()).all()
    
    csv_path = f'temp_attendance_{int(time.time())}.csv'
    try:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([f'Attendance Report for {current_user.class_name}'])
            writer.writerow(['ID', 'Student Name', 'Class Name', 'Date', 'Time'])
            writer.writerows(records)
        
        from_email = 'shreyagaikwad2710@gmail.com'
        password = 'jolmpqlfusjhflis'
        msg = MIMEMultipart()
        msg['From'] = from_email
        msg['To'] = recipient_email
        msg['Subject'] = f'Attendance Report for {current_user.class_name}'
        msg.attach(MIMEText('Attached is the attendance report.', 'plain'))
        
        with open(csv_path, 'rb') as attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename={csv_path}')
            msg.attach(part)
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_email, password)
        server.sendmail(from_email, recipient_email, msg.as_string())
        server.quit()
        return jsonify({'message': 'Email sent successfully to your registered email!'})
    except Exception as e:
        return jsonify({'message': f'Error sending email: {str(e)}'})
    finally:
        if os.path.exists(csv_path):
            os.remove(csv_path)


@app.route('/student_panel')
def student_panel():
    return render_template('student_panel.html')

@app.route('/mark_attendance_student', methods=['POST'])
def mark_attendance_student():
    try:
        image_data = request.form['image']
        if not image_data:
            return jsonify({'status': 'error', 'message': 'No image data received.'})
        
        import base64
        image = base64.b64decode(image_data.split(',')[1])
        np_arr = np.frombuffer(image, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'status': 'error', 'message': 'Invalid image format.'})
        
        # Use face_recognition
        face_encodings = face_recognition.face_encodings(img)
        if not face_encodings:
            return jsonify({'status': 'error', 'message': 'No face detected.'})
        
        for face_encoding in face_encodings:
            distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(distances)
            best_distance = distances[best_match_index]
            
            if best_distance < 0.6:
                student_id = known_face_student_ids[best_match_index]
                now = datetime.now()
                date = now.strftime('%Y-%m-%d')
                time_str = now.strftime('%H-%M-%S')
                
                student = Student.query.filter_by(id=student_id).first()
                if student:
                    student_name, student_class = student.name, student.class_name
                    existing = Attendance.query.filter_by(student_id=student_id, date=date).first()
                    if not existing:
                        new_id = db.session.query(db.func.max(Attendance.id)).scalar() or 0 + 1
                        new_attendance = Attendance(id=new_id, student_id=student_id, date=date, time=time_str)
                        db.session.add(new_attendance)
                        db.session.commit()
                        return jsonify({'status': 'success', 'message': f'Attendance marked for {student_name} ({student_class})!'})
                    return jsonify({'status': 'info', 'message': 'Already marked today.'})
        
        return jsonify({'status': 'error', 'message': 'Face not recognized.'})
    
    except Exception as e:
        app.logger.error(f"Error: {e}")
        return jsonify({'status': 'error', 'message': 'Error processing.'})
    
@app.route('/qr_code')
@login_required
def qr_code():
    app_url = 'https://your-render-app-name.onrender.com/'  # Replace with your Render URL
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(app_url)
    qr.make(fit=True)
    img = qr.make_image(fill='black', back_color='white')
    buf = BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return Response(buf.getvalue(), mimetype='image/png', headers={'Content-Disposition': 'attachment; filename=attendance_qr.png'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)