from flask import Flask, render_template, request, redirect, url_for, flash, send_file, Response
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///attendance.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Models
class Admin(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    class_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100))

class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    class_name = db.Column(db.String(100), nullable=False)
    class_display_id = db.Column(db.String(20))
    encoding = db.Column(db.LargeBinary)
    enrolled = db.Column(db.Boolean, default=False)

class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'), nullable=False)
    date = db.Column(db.String(20), nullable=False)
    time = db.Column(db.String(20), nullable=False)

class KioskStatus(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    active = db.Column(db.Boolean, default=False)
    admin_info = db.Column(db.String(100))

@login_manager.user_loader
def load_user(user_id):
    return Admin.query.get(int(user_id))

# Routes
@app.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        admin = Admin.query.filter_by(username=username).first()
        
        if admin and check_password_hash(admin.password, password):
            login_user(admin)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password!', 'danger')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])
        class_name = request.form['class_name']
        email = request.form.get('email', '')
        
        if Admin.query.filter_by(username=username).first():
            flash('Username already exists!', 'danger')
            return redirect(url_for('register'))
        
        new_admin = Admin(username=username, password=password, class_name=class_name, email=email)
        db.session.add(new_admin)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully!', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    total_students = Student.query.filter_by(class_name=current_user.class_name).count()
    today = datetime.now().strftime('%Y-%m-%d')
    todays_count = Attendance.query.join(Student).filter(
        Student.class_name == current_user.class_name,
        Attendance.date == today
    ).count()
    
    kiosk = KioskStatus.query.first()
    kiosk_active = kiosk.active if kiosk else False
    kiosk_admin = kiosk.admin_info if kiosk else None
    
    return render_template('dashboard.html', 
                         total_students=total_students,
                         todays_count=todays_count,
                         kiosk_active=kiosk_active,
                         kiosk_admin=kiosk_admin)

@app.route('/students', methods=['GET', 'POST'])
@login_required
def students():
    if request.method == 'POST':
        name = request.form['name']
        class_name = current_user.class_name
        
        # Get next display ID
        last_student = Student.query.filter_by(class_name=class_name).order_by(Student.id.desc()).first()
        new_id = (last_student.class_display_id if last_student else '0')
        new_num = int(new_id) + 1 if new_id.isdigit() else 1
        class_display_id = str(new_num).zfill(3)
        
        new_student = Student(name=name, class_name=class_name, class_display_id=class_display_id)
        db.session.add(new_student)
        db.session.commit()
        flash('Student added! Now enroll face.', 'success')
        return redirect(url_for('enroll_face', id=new_student.id))
    
    students_list = Student.query.filter_by(class_name=current_user.class_name).all()
    return render_template('students.html', students=students_list)

@app.route('/enroll_face/<int:id>')
@login_required
def enroll_face(id):
    student = Student.query.get_or_404(id)
    if student.class_name != current_user.class_name:
        flash('Access denied!', 'danger')
        return redirect(url_for('students'))
    return render_template('enroll_face.html', student=student)

@app.route('/capture_face/<int:id>', methods=['POST'])
@login_required
def capture_face(id):
    student = Student.query.get_or_404(id)
    if student.class_name != current_user.class_name:
        return {'status': 'error', 'message': 'Access denied!'}
    
    data = request.get_json()
    image_data = data.get('image', '')
    
    try:
        header, encoded = image_data.split(',', 1)
        image_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            orb = cv2.ORB_create()
            kp, des = orb.detectAndCompute(face_roi, None)
            
            if des is not None:
                student.encoding = des.tobytes()
                student.enrolled = True
                db.session.commit()
                return {'status': 'success', 'message': 'Face enrolled successfully!'}
        
        return {'status': 'error', 'message': 'No face detected. Try again!'}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

@app.route('/delete_student/<int:id>', methods=['POST'])
@login_required
def delete_student(id):
    student = Student.query.get_or_404(id)
    if student.class_name != current_user.class_name:
        flash('Access denied!', 'danger')
        return redirect(url_for('students'))
    
    Attendance.query.filter_by(student_id=id).delete()
    db.session.delete(student)
    db.session.commit()
    flash('Student deleted!', 'success')
    return redirect(url_for('students'))

@app.route('/attendance')
@login_required
def attendance():
    date_filter = request.args.get('date', '')
    search = request.args.get('search', '')
    
    query = Attendance.query.join(Student).filter(Student.class_name == current_user.class_name)
    
    if date_filter:
        query = query.filter(Attendance.date == date_filter)
    
    if search:
        query = query.filter(Student.name.ilike(f'%{search}%'))
    
    records = query.order_by(Attendance.id.desc()).all()
    return render_template('attendance.html', records=records)

@app.route('/export_attendance_csv')
@login_required
def export_attendance_csv():
    date_filter = request.args.get('date', '')
    search = request.args.get('search', '')
    
    query = Attendance.query.join(Student).filter(Student.class_name == current_user.class_name)
    
    if date_filter:
        query = query.filter(Attendance.date == date_filter)
    if search:
        query = query.filter(Student.name.ilike(f'%{search}%'))
    
    records = query.all()
    
    csv_data = 'ID,Name,Date,Time\n'
    for r in records:
        student = Student.query.get(r.student_id)
        csv_data += f'{student.class_display_id},{student.name},{r.date},{r.time}\n'
    
    return Response(csv_data, mimetype='text/csv', headers={'Content-Disposition': 'attachment;filename=attendance.csv'})

@app.route('/kiosk_status', methods=['GET', 'POST', 'DELETE'])
@login_required
def kiosk_status():
    kiosk = KioskStatus.query.first()
    if not kiosk:
        kiosk = KioskStatus()
        db.session.add(kiosk)
        db.session.commit()
    
    if request.method == 'GET':
        return {'active': kiosk.active, 'admin': kiosk.admin_info}
    
    if request.method == 'POST':
        if kiosk.active:
            return {'status': 'already_active', 'message': f'Kiosk already active by {kiosk.admin_info}'}
        kiosk.active = True
        kiosk.admin_info = current_user.username
        db.session.commit()
        return {'status': 'active', 'message': 'Kiosk started!'}
    
    if request.method == 'DELETE':
        kiosk.active = False
        kiosk.admin_info = ''
        db.session.commit()
        return {'status': 'success', 'message': 'Kiosk stopped!'}

@app.route('/kiosk_display')
def kiosk_display():
    return render_template('kiosk_display.html')

@app.route('/mark_attendance_student', methods=['POST'])
def mark_attendance_student():
    try:
        data = request.get_json()
        image_data = data.get('image', '')
        
        header, encoded = image_data.split(',', 1)
        image_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {'status': 'error', 'message': 'Invalid image'}
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        if len(faces) == 0:
            return {'status': 'error', 'message': 'No face detected'}
        
        x, y, w, h = faces[0]
        face_roi = gray[y:y+h, x:x+w]
        
        orb = cv2.ORB_create()
        kp, des = orb.detectAndCompute(face_roi, None)
        
        if des is None:
            return {'status': 'error', 'message': 'Face features not clear'}
        
        students = Student.query.filter(Student.enrolled == True).all()
        best_match = None
        best_score = 0
        
        for student in students:
            if student.encoding:
                known_des = np.frombuffer(student.encoding, np.float64)
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(des, known_des)
                score = len(matches)
                
                if score > best_score and score > 15:
                    best_score = score
                    best_match = student
        
        if not best_match:
            return {'status': 'error', 'message': 'Face not recognized'}
        
        today = datetime.now().strftime('%Y-%m-%d')
        time_now = datetime.now().strftime('%H:%M:%S')
        
        if Attendance.query.filter_by(student_id=best_match.id, date=today).first():
            return {'status': 'info', 'message': 'Already marked today'}
        
        new_attendance = Attendance(student_id=best_match.id, date=today, time=time_now)
        db.session.add(new_attendance)
        db.session.commit()
        
        return {'status': 'success', 'message': f'Attendance marked for {best_match.name}!'}
    
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

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
            Attendance.date >= week_start_str,
            Attendance.date <= week_end_str
        ).count()
        
        percentage = (attendance_count / 5) * 100 if 5 > 0 else 0
        
        if percentage < 50:
            blacklisted_students.append({
                'id': student.id,
                'name': student.name,
                'days': attendance_count,
                'percentage': round(percentage, 1)
            })
    
    return render_template('blacklist.html', 
                         blacklisted_students=blacklisted_students,
                         week_start=week_start_str,
                         week_end=week_end_str)

@app.route('/insights')
@login_required
def insights():
    total_records = Attendance.query.join(Student).filter(
        Student.class_name == current_user.class_name
    ).count()
    
    unique_students = db.session.query(Attendance.student_id).join(Student).filter(
        Student.class_name == current_user.class_name
    ).distinct().count()
    
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

@app.route('/qr_code')
@login_required
def qr_code():
    import qrcode
    from io import BytesIO
    
    url = request.host_url.rstrip('/') + '/kiosk_display'
    img = qrcode.make(url)
    buf = BytesIO()
    img.save(buf, 'PNG')
    buf.seek(0)
    
    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=port)