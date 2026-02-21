from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import cv2  # For OpenCV face detection
import numpy as np  # For numpy arrays
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
import base64  # For image encoding

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

# Global variables for face recognition (using ORB descriptors)
known_face_encodings = []  # List of ORB descriptors
known_face_student_ids = []

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
    encoding = db.Column(db.Text)  # Stores base64 image
    class_display_id = db.Column(db.Integer)

class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'))
    date = db.Column(db.String(20))
    time = db.Column(db.String(20))
    student = db.relationship('Student', backref='attendance_records')

class KioskStatus(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    active = db.Column(db.Boolean, default=False)
    admin_info = db.Column(db.String(200))
    updated_at = db.Column(db.DateTime, default=datetime.now)

@login_manager.user_loader
def load_user(user_id):
    return Admin.query.get(int(user_id))

# Load known faces (ORB features)
def load_known_faces():
    global known_face_encodings, known_face_student_ids
    known_face_encodings = []
    known_face_student_ids = []
    students = Student.query.filter(Student.encoding.isnot(None)).all()
    orb = cv2.ORB_create()
    for student in students:
        img_data = base64.b64decode(student.encoding)
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_GRAYSCALE)
        kp, des = orb.detectAndCompute(img, None)
        known_face_encodings.append(des)
        known_face_student_ids.append(student.id)
    app.logger.info(f"Loaded {len(known_face_encodings)} face features.")

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

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])
        class_name = request.form['class_name']
        email = request.form['email']
        
        # Check if username exists
        existing_admin = Admin.query.filter_by(username=username).first()
        if existing_admin:
            flash('Username already exists!', 'danger')
            return redirect(url_for('register'))
        
        # Create new admin WITHOUT manual id
        new_admin = Admin(
            username=username,
            password=password,
            class_name=class_name,
            email=email
        )
        db.session.add(new_admin)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')
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
    # Count total students for this admin's class
    total_students = Student.query.filter_by(class_name=current_user.class_name).count()
    
    # Count today's attendance for this admin's class
    today = datetime.now().strftime('%Y-%m-%d')
    todays_count = db.session.query(Attendance).join(Student).filter(
        Student.class_name == current_user.class_name,
        Attendance.date == today
    ).count()
    
    # Check kiosk status
    kiosk = KioskStatus.query.first()
    kiosk_active = kiosk.active if kiosk else False
    kiosk_admin = kiosk.admin_info if kiosk else ''
    
    return render_template('dashboard.html', 
                          total_students=total_students, 
                          todays_count=todays_count,
                          kiosk_active=kiosk_active,
                          kiosk_admin=kiosk_admin)

@app.route('/kiosk_display')
def kiosk_display():
    return render_template('kiosk_display.html')

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
            image = base64.b64decode(image_data.split(',')[1])
            img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
            
            # Detect face with OpenCV
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                # Store the full image for feature matching
                encoding_str = base64.b64encode(image).decode('utf-8')
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

@app.route('/mark_attendance_student', methods=['POST'])
def mark_attendance_student():
    load_known_faces()
    try:
        if request.is_json:
            data = request.get_json()
            image_data = data.get('image')
        else:
            image_data = request.form.get('image')

        if not image_data:
            return jsonify({'status': 'error', 'message': 'No image received.'})

        image_bytes = base64.b64decode(image_data.split(',')[1])
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'status': 'error', 'message': 'Invalid image.'})

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        if len(faces) == 0:
            return jsonify({'status': 'error', 'message': 'No face detected.'})

        x, y, w, h = faces[0]
        face_roi = gray[y:y+h, x:x+w]

        orb = cv2.ORB_create()
        kp, des = orb.detectAndCompute(face_roi, None)

        if des is None:
            return jsonify({'status': 'error', 'message': 'Face features not clear.'})

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        best_match_id = None
        best_score = 0

        for i, known_des in enumerate(known_face_encodings):
            if known_des is not None:
                matches = bf.match(des, known_des)
                score = len(matches)

                if score > best_score and score > 15:
                    best_score = score
                    best_match_id = known_face_student_ids[i]

        if not best_match_id:
            return jsonify({'status': 'error', 'message': 'Face not recognized.'})

        student = Student.query.filter_by(id=best_match_id).first()
        if not student:
            return jsonify({'status': 'error', 'message': 'Student not found.'})

        today = datetime.now().strftime('%Y-%m-%d')
        
        # FIX: Use time() instead of strftime for proper time storage
        now = datetime.now()
        time_now = now.strftime('%H:%M:%S')

        already_marked = Attendance.query.filter_by(
            student_id=best_match_id,
            date=today
        ).first()

        if already_marked:
            return jsonify({'status': 'info', 'message': 'Attendance already marked today.'})

        new_attendance = Attendance(
            student_id=best_match_id,
            date=today,
            time=time_now
        )
        db.session.add(new_attendance)
        db.session.commit()

        return jsonify({
            'status': 'success',
            'message': f'Attendance marked for {student.name}'
        })

    except Exception as e:
        app.logger.error(e)
        return jsonify({'status': 'error', 'message': 'Server error.'})

@app.route('/insights')
@login_required
def insights():
    try:
        # Total records
        total_records = Attendance.query.join(Student).filter(
            Student.class_name == current_user.class_name
        ).count()
        
        # Unique students
        unique_students = db.session.query(Attendance.student_id).join(Student).filter(
            Student.class_name == current_user.class_name
        ).distinct().count()
        
        # Attendance per student
        attendance_per_student = db.session.query(
            Student.name,
            db.func.count(Attendance.id)
        ).join(Attendance, Student.id == Attendance.student_id).filter(
            Student.class_name == current_user.class_name
        ).group_by(Student.id, Student.name).all()
        
        insights_data = {
            'total_records': total_records,
            'unique_students': unique_students,
            'attendance_per_student': attendance_per_student
        }
        
        return render_template('insights.html', insights_data=insights_data)
    
    except Exception as e:
        app.logger.error(f"Insights error: {e}")
        return render_template('insights.html', insights_data={
            'total_records': 0,
            'unique_students': 0,
            'attendance_per_student': []
        })
    
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
        attendance_count = Attendance.query.filter(
            Attendance.student_id == student.id,
            Attendance.date.between(week_start_str, week_end_str)
        ).count()
        
        percentage = (attendance_count / 5) * 100 if 5 > 0 else 0
        
        if percentage < 50:
            blacklisted_students.append({
                'id': student.id,
                'name': student.name,
                'days': attendance_count,
                'percentage': round(percentage, 1)
            })
    
    return render_template(
        'blacklist.html', 
        blacklisted_students=blacklisted_students, 
        week_start=week_start_str, 
        week_end=week_end_str
    )

@app.route('/kiosk_status', methods=['GET', 'POST', 'DELETE'])
@login_required
def kiosk_status():
    kiosk = KioskStatus.query.first()
    
    if request.method == 'POST':
        # Check if already active
        if kiosk and kiosk.active:
            return jsonify({
                'status': 'already_active', 
                'message': f'Kiosk is already running! Started by: {kiosk.admin_info}'
            })
        
        # Start kiosk
        if not kiosk:
            kiosk = KioskStatus(active=True, admin_info=f"{current_user.username} ({current_user.class_name})")
            db.session.add(kiosk)
        else:
            kiosk.active = True
            kiosk.admin_info = f"{current_user.username} ({current_user.class_name})"
            kiosk.updated_at = datetime.now()
        db.session.commit()
        return jsonify({'status': 'active', 'message': 'Kiosk started successfully!'})
    
    elif request.method == 'DELETE':
        if kiosk:
            kiosk.active = False
            kiosk.admin_info = ''
            kiosk.updated_at = datetime.now()
            db.session.commit()
        return jsonify({'status': 'inactive', 'message': 'Kiosk stopped!'})
    
    else:
        if kiosk and kiosk.active:
            return jsonify({'active': True, 'admin': kiosk.admin_info})
        return jsonify({'active': False, 'admin': ''})

@app.route('/export_attendance_csv')
@login_required
def export_attendance_csv():
    records = db.session.query(Attendance.id, Student.name, Attendance.date, Attendance.time).join(Student).filter(Student.class_name == current_user.class_name).order_by(Attendance.date.desc(), Attendance.time.desc()).all()
    
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow([f'Attendance Report for {current_user.class_name}'])
    writer.writerow(['ID', 'Student Name', 'Date', 'Time'])
    for record in records:
        writer.writerow([record[0], record[1], f'"{record[2]}"', record[3]])
    
    output.seek(0)
    return Response(output.getvalue(), mimetype='text/csv', headers={'Content-Disposition': 'attachment; filename=attendance.csv'})


@app.route('/qr_code')
@login_required
def qr_code():
    app_url = 'https://faceattendance-production-f80c.up.railway.app'  # Replace with your Railway URL
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
   
