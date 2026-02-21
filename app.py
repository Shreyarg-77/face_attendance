from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, EmailField
from wtforms.validators import DataRequired, Length, Regexp
from werkzeug.security import generate_password_hash, check_password_hash
import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime, timedelta
import csv
from io import StringIO, BytesIO
import qrcode
import logging
import base64
import secrets
import threading
from functools import wraps

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# SECURITY FIX 1: Generate proper secret key
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(32))

# Database configuration
DATABASE_URL = os.environ.get('DATABASE_URL')
if DATABASE_URL:
    app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///models/attendance.db'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_size': 5,
    'pool_recycle': 300,
    'pool_pre_ping': True,
    'pool_timeout': 30
}


db = SQLAlchemy(app)

# Thread-safe face encoding cache
class FaceEncodingCache:
    def __init__(self):
        self.encodings = []
        self.student_ids = []
        self.lock = threading.Lock()
        self.last_updated = None
    
    def update(self, encodings, student_ids):
        with self.lock:
            self.encodings = encodings
            self.student_ids = student_ids
            self.last_updated = datetime.now()
    
    def get_data(self):
        with self.lock:
            return self.encodings.copy(), self.student_ids.copy()
    
    def get_encoding(self, index):
        with self.lock:
            if 0 <= index < len(self.encodings):
                return self.encodings[index]
            return None
    
    def get_student_id(self, index):
        with self.lock:
            if 0 <= index < len(self.student_ids):
                return self.student_ids[index]
            return None
    
    def get_length(self):
        with self.lock:
            return len(self.encodings)

face_cache = FaceEncodingCache()

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.session_protection = "strong"

# Ensure models folder exists
os.makedirs('models', exist_ok=True)

# Download Haar Cascade for production
cascade_path = 'models/haarcascade_frontalface_default.xml'
if not os.path.exists(cascade_path):
    try:
        import urllib.request
        url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
        urllib.request.urlretrieve(url, cascade_path)
        logger.info("Haar cascade downloaded successfully")
    except Exception as e:
        logger.warning(f"Could not download cascade: {e}")

# Models
class Admin(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    class_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.now)

class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    class_name = db.Column(db.String(100), nullable=False)
    class_display_id = db.Column(db.String(20))
    encoding = db.Column(db.LargeBinary)  # 128-dimensional face encoding
    enrolled = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.now)

class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id', ondelete='CASCADE'))
    date = db.Column(db.String(20), nullable=False)
    time = db.Column(db.String(20), nullable=False)
    student = db.relationship('Student', backref='attendance_records')

class KioskStatus(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    active = db.Column(db.Boolean, default=False)
    admin_info = db.Column(db.String(200))
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

# Add indexes for performance
db.Index('idx_attendance_date', Attendance.date)
db.Index('idx_attendance_student_date', Attendance.student_id, Attendance.date)
db.Index('idx_student_class', Student.class_name)

@login_manager.user_loader
def load_user(user_id):
    return Admin.query.get(int(user_id))

# WTForms for CSRF protection and validation
class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[
        DataRequired(),
        Length(min=3, max=100),
        Regexp(r'^[a-zA-Z0-9_]+$', message='Username can only contain letters, numbers, and underscores')
    ])
    password = PasswordField('Password', validators=[
        DataRequired(),
        Length(min=6, max=100)
    ])
    class_name = StringField('Class Name', validators=[
        DataRequired(),
        Length(min=2, max=100)
    ])
    email = EmailField('Email', validators=[])

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])

class StudentForm(FlaskForm):
    name = StringField('Name', validators=[
        DataRequired(),
        Length(min=2, max=100),
        Regexp(r'^[a-zA-Z\s]+$', message='Name can only contain letters and spaces')
    ])

def load_face_encodings_thread_safe():
    """Thread-safe function to load face encodings from database"""
    try:
        students = Student.query.filter(
            Student.encoding.isnot(None),
            Student.enrolled == True
        ).all()
        
        encodings = []
        student_ids = []
        
        for student in students:
            try:
                encoding_data = student.encoding
                
                # Handle both bytes and string encodings
                if isinstance(encoding_data, str):
                    encoding_array = np.frombuffer(base64.b64decode(encoding_data), dtype=np.float64)
                else:
                    encoding_array = np.frombuffer(encoding_data, dtype=np.float64)
                
                # Ensure it's the correct shape (128 dimensions)
                if len(encoding_array) == 128:
                    encodings.append(encoding_array)
                    student_ids.append(student.id)
                    
            except Exception as e:
                logger.warning(f"Could not load encoding for student {student.id}: {e}")
                continue
        
        face_cache.update(encodings, student_ids)
        logger.info(f"Loaded {len(encodings)} face encodings")
        
    except Exception as e:
        logger.error(f"Error loading face encodings: {e}")

# Initialize database and load faces
with app.app_context():
    try:
        db.create_all()
        load_face_encodings_thread_safe()
        logger.info("App initialized successfully")
    except Exception as e:
        logger.error(f"Startup error: {e}")

# Routes
@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        class_name = form.class_name.data
        email = form.email.data if form.email.data else ''
        
        # Check if username exists
        existing_admin = Admin.query.filter_by(username=username).first()
        if existing_admin:
            flash('Username already exists!', 'danger')
            return redirect(url_for('register'))
        
        new_admin = Admin(
            username=username,
            password=generate_password_hash(password),
            class_name=class_name,
            email=email
        )
        
        db.session.add(new_admin)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    # Handle form errors
    if form.errors:
        for field, errors in form.errors.items():
            for error in errors:
                flash(f'{field}: {error}', 'danger')
    
    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return render_template('login.html', already_logged_in=True)
    
    form = LoginForm()
    
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        
        admin = Admin.query.filter_by(username=username).first()
        
        if admin and check_password_hash(admin.password, password):
            login_user(admin)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password!', 'danger')
    
    return render_template('login.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    total_students = Student.query.filter_by(class_name=current_user.class_name).count()
    today = datetime.now().strftime('%Y-%m-%d')
    todays_count = db.session.query(Attendance).join(Student).filter(
        Student.class_name == current_user.class_name,
        Attendance.date == today
    ).count()
    
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
    
    # SECURITY FIX: Sanitize input
    name_query = request.args.get('name', '').strip()
    date_query = request.args.get('date', today if view == 'today' else '').strip()
    
    # Validate date format
    try:
        if date_query:
            datetime.strptime(date_query, '%Y-%m-%d')
    except ValueError:
        date_query = today
        flash('Invalid date format. Using today.', 'warning')
    
    if view == 'history':
        date_records = db.session.query(
            Attendance.date, 
            db.func.count(Attendance.id)
        ).join(Student).filter(
            Student.class_name == current_user.class_name
        ).group_by(Attendance.date).order_by(Attendance.date.desc()).all()
        return render_template('attendance.html', view=view, date_records=date_records)
    
    elif view == 'date' and selected_date:
        query = Attendance.query.join(Student).filter(
            Student.class_name == current_user.class_name, 
            Attendance.date == selected_date
        )
        
        # SECURITY FIX: Use parameterized queries
        if name_query:
            query = query.filter(Student.name.ilike(f'%{name_query}%'))
        
        records = query.order_by(Attendance.time.desc()).all()
        return render_template('attendance.html', view=view, records=records, selected_date=selected_date)
    
    else:
        query = Attendance.query.join(Student).filter(
            Student.class_name == current_user.class_name, 
            Attendance.date == date_query
        )
        
        if name_query:
            query = query.filter(Student.name.ilike(f'%{name_query}%'))
        
        records = query.order_by(Attendance.time.desc()).all()
        return render_template('attendance.html', view='today', records=records, selected_date=date_query)

@app.route('/students', methods=['GET', 'POST'])
@login_required
def students():
    form = StudentForm()
    
    if form.validate_on_submit():
        name = form.name.data
        
        max_display_id = db.session.query(
            db.func.max(Student.class_display_id)
        ).filter(Student.class_name == current_user.class_name).scalar() or 0
        
        class_display_id = max_display_id + 1
        new_id = (db.session.query(db.func.max(Student.id)).scalar() or 0) + 1
        
        new_student = Student(
            id=new_id,
            name=name,
            class_name=current_user.class_name,
            class_display_id=class_display_id
        )
        
        db.session.add(new_student)
        db.session.commit()
        flash('Student added successfully!', 'success')
        return redirect(url_for('students'))
    
    students_list = Student.query.filter_by(
        class_name=current_user.class_name
    ).order_by(Student.class_display_id).all()
    
    encodings, student_ids = face_cache.get_data()
    enrolled_status = {sid: True for sid in student_ids}
    
    return render_template('students.html', students=students_list, enrolled_status=enrolled_status, form=form)

@app.route('/delete_student/<int:student_id>', methods=['POST'])
@login_required
def delete_student(student_id):
    student = Student.query.filter_by(
        id=student_id, 
        class_name=current_user.class_name
    ).first()
    
    if not student:
        flash('Student not found or not in your class.', 'danger')
        return redirect(url_for('students'))
    
    try:
        # Delete attendance records first
        Attendance.query.filter_by(student_id=student_id).delete()
        
        # Delete student
        db.session.delete(student)
        db.session.commit()
        
        # Reorder class_display_id
        remaining_students = Student.query.filter_by(
            class_name=current_user.class_name
        ).order_by(Student.class_display_id).all()
        
        for i, student in enumerate(remaining_students, start=1):
            student.class_display_id = str(i)
        
        db.session.commit()
        
        # Reload face cache
        load_face_encodings_thread_safe()
        
        flash(f'Student {student.name} deleted successfully.', 'success')
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error deleting student: {e}")
        flash('Error deleting student.', 'danger')
    
    return redirect(url_for('students'))

@app.route('/edit_student/<int:student_id>', methods=['GET', 'POST'])
@login_required
def edit_student(student_id):
    student = Student.query.filter_by(
        id=student_id, 
        class_name=current_user.class_name
    ).first()
    
    if not student:
        flash('Student not found or not in your class.', 'danger')
        return redirect(url_for('students'))
    
    form = StudentForm()
    
    if form.validate_on_submit():
        student.name = form.name.data
        db.session.commit()
        flash('Student updated successfully!', 'success')
        return redirect(url_for('students'))
    
    form.name.data = student.name
    return render_template('edit_student.html', student=student, form=form)

@app.route('/enroll_face/<int:student_id>', methods=['GET', 'POST'])
@login_required
def enroll_face(student_id):
    student = Student.query.filter_by(
        id=student_id, 
        class_name=current_user.class_name
    ).first()
    
    if not student:
        flash('Student not found or not in your class.', 'danger')
        return redirect(url_for('students'))
    
    if request.method == 'POST':
        image_data = request.form.get('image')
        
        if not image_data:
            flash('No image received.', 'danger')
            return redirect(url_for('students'))
        
        try:
            # Extract base64 data
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            
            if img is None:
                flash('Invalid image.', 'danger')
                return redirect(url_for('students'))
            
            # Convert to RGB for face_recognition library
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Detect faces using face_recognition library
            face_locations = face_recognition.face_locations(rgb_img)
            
            if len(face_locations) == 0:
                flash('No face detected. Please try again with better lighting.', 'danger')
                return redirect(url_for('students'))
            
            # Get face encoding (128-dimensional)
            face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
            
            if len(face_encodings) == 0:
                flash('Could not extract face features. Please try again.', 'danger')
                return redirect(url_for('students'))
            
            # Store encoding as bytes
            encoding_bytes = face_encodings[0].tobytes()
            student.encoding = encoding_bytes
            student.enrolled = True
            db.session.commit()
            
            # Reload face cache
            load_face_encodings_thread_safe()
            
            flash(f'Face enrolled successfully for {student.name}!', 'success')
            return redirect(url_for('students'))
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            flash('Error processing image. Please try again.', 'danger')
    
    return render_template('enroll_face.html', student=student)

@app.route('/mark_attendance_student', methods=['POST'])
def mark_attendance_student():
    """Mark attendance using face recognition"""
    try:
        # Get image data
        if request.is_json:
            data = request.get_json()
            image_data = data.get('image')
        else:
            image_data = request.form.get('image')
        
        if not image_data:
            return jsonify({'status': 'error', 'message': 'No image received.'})
        
        # Extract base64 data
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'status': 'error', 'message': 'Invalid image.'})
        
        # Convert to RGB
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_img)
        
        if len(face_locations) == 0:
            return jsonify({'status': 'error', 'message': 'No face detected.'})
        
        # Get face encodings
        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
        
        if len(face_encodings) == 0:
            return jsonify({'status': 'error', 'message': 'Face features not clear.'})
        
        # Compare with known faces
        unknown_encoding = face_encodings[0]
        encodings, student_ids = face_cache.get_data()
        
        if len(encodings) == 0:
            return jsonify({'status': 'error', 'message': 'No enrolled faces in database.'})
        
        # Compare faces using proper face recognition
        matches = face_recognition.compare_faces(encodings, unknown_encoding, tolerance=0.5)
        face_distances = face_recognition.face_distance(encodings, unknown_encoding)
        
        if True in matches:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                student_id = student_ids[best_match_index]
                
                student = Student.query.get(student_id)
                if not student:
                    return jsonify({'status': 'error', 'message': 'Student not found.'})
                
                today = datetime.now().strftime('%Y-%m-%d')
                time_now = datetime.now().strftime('%H:%M:%S')
                
                # Check if already marked today
                already_marked = Attendance.query.filter_by(
                    student_id=student_id,
                    date=today
                ).first()
                
                if already_marked:
                    return jsonify({
                        'status': 'info', 
                        'message': f'Attendance already marked for {student.name} today.'
                    })
                
                # Mark attendance
                new_attendance = Attendance(
                    student_id=student_id,
                    date=today,
                    time=time_now
                )
                db.session.add(new_attendance)
                db.session.commit()
                
                return jsonify({
                    'status': 'success',
                    'message': f'Attendance marked for {student.name}'
                })
        
        return jsonify({'status': 'error', 'message': 'Face not recognized.'})
        
    except Exception as e:
        logger.error(f"Error in mark_attendance_student: {e}")
        return jsonify({'status': 'error', 'message': 'Server error.'})

@app.route('/insights')
@login_required
def insights():
    try:
        # Total records
        total_records = Attendance.query.join(Student).filter(
            Student.class_name == current_user.class_name
        ).count()
        
        # Unique students with attendance
        unique_students = db.session.query(
            Attendance.student_id
        ).join(Student).filter(
            Student.class_name == current_user.class_name
        ).distinct().count()
        
        # Attendance per student
        attendance_per_student = db.session.query(
            Student.name,
            db.func.count(Attendance.id)
        ).join(Attendance, Student.id == Attendance.student_id).filter(
            Student.class_name == current_user.class_name
        ).group_by(Student.id, Student.name).order_by(
            db.func.count(Attendance.id).desc()
        ).all()
        
        # Weekly stats
        today = datetime.now()
        week_start = today - timedelta(days=today.weekday())
        week_end = week_start + timedelta(days=6)
        
        weekly_attendance = db.session.query(
            db.func.date(Attendance.date),
            db.func.count(Attendance.id)
        ).join(Student).filter(
            Student.class_name == current_user.class_name,
            Attendance.date.between(
                week_start.strftime('%Y-%m-%d'),
                week_end.strftime('%Y-%m-%d')
            )
        ).group_by(db.func.date(Attendance.date)).all()
        
        insights_data = {
            'total_records': total_records,
            'unique_students': unique_students,
            'attendance_per_student': attendance_per_student,
            'weekly_attendance': weekly_attendance,
            'week_start': week_start.strftime('%Y-%m-%d'),
            'week_end': week_end.strftime('%Y-%m-%d')
        }
        
        return render_template('insights.html', insights_data=insights_data)
        
    except Exception as e:
        logger.error(f"Insights error: {e}")
        return render_template('insights.html', insights_data={
            'total_records': 0,
            'unique_students': 0,
            'attendance_per_student': [],
            'weekly_attendance': [],
            'week_start': '',
            'week_end': ''
        })

@app.route('/blacklist')
@login_required
def blacklist():
    today = datetime.now()
    week_start = today - timedelta(days=today.weekday())
    week_end = week_start + timedelta(days=6)
    week_start_str = week_start.strftime('%Y-%m-%d')
    week_end_str = week_end.strftime('%Y-%m-%d')
    
    # Get all students in class
    students_list = Student.query.filter_by(class_name=current_user.class_name).all()
    blacklisted_students = []
    
    for student in students_list:
        attendance_count = Attendance.query.filter(
            Attendance.student_id == student.id,
            Attendance.date.between(week_start_str, week_end_str)
        ).count()
        
        # Assuming 5 working days per week
        percentage = (attendance_count / 5) * 100 if attendance_count <= 5 else 100
        
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
            kiosk = KioskStatus(
                active=True, 
                admin_info=f"{current_user.username} ({current_user.class_name})"
            )
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
    try:
        records = db.session.query(
            Attendance.id, 
            Student.name, 
            Attendance.date, 
            Attendance.time
        ).join(Student).filter(
            Student.class_name == current_user.class_name
        ).order_by(
            Attendance.date.desc(), 
            Attendance.time.desc()
        ).all()
        
        output = StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([f'Attendance Report for {current_user.class_name}'])
        writer.writerow([f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'])
        writer.writerow([])
        writer.writerow(['ID', 'Student Name', 'Date', 'Time'])
        
        # Write data
        for record in records:
            writer.writerow([record[0], record[1], record[2], record[3]])
        
        output.seek(0)
        
        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={
                'Content-Disposition': f'attachment; filename=attendance_{current_user.class_name}_{datetime.now().strftime("%Y%m%d")}.csv'
            }
        )
        
    except Exception as e:
        logger.error(f"Error exporting CSV: {e}")
        flash('Error exporting attendance data.', 'danger')
        return redirect(url_for('attendance'))

@app.route('/qr_code')
@login_required
def qr_code():
    try:
        # Get base URL from request
        base_url = request.host_url.rstrip('/')
        app_url = f"{base_url}/kiosk_display"
        
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(app_url)
        qr.make(fit=True)
        
        img = qr.make_image(fill='black', back_color='white')
        buf = BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        
        return Response(
            buf.getvalue(),
            mimetype='image/png',
            headers={
                'Content-Disposition': 'attachment; filename=attendance_qr.png'
            }
        )
        
    except Exception as e:
        logger.error(f"Error generating QR code: {e}")
        flash('Error generating QR code.', 'danger')
        return redirect(url_for('dashboard'))

@app.route('/refresh_faces', methods=['POST'])
@login_required
def refresh_faces():
    """Manually refresh face encodings cache"""
    try:
        load_face_encodings_thread_safe()
        count = face_cache.get_length()
        return jsonify({
            'status': 'success', 
            'message': f'Face cache refreshed. {count} faces loaded.'
        })
    except Exception as e:
        logger.error(f"Error refreshing faces: {e}")
        return jsonify({'status': 'error', 'message': 'Error refreshing face cache.'})

@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', error='Page not found'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('error.html', error='Internal server error'), 500

@app.errorhandler(403)
def forbidden_error(error):
    return render_template('error.html', error='Access forbidden'), 403

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)