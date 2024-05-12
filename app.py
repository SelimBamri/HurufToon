import os
import random
from datetime import datetime
from urllib import request
from flask import Flask, render_template, redirect, url_for, session, request
from flask_sqlalchemy import SQLAlchemy
import pickle as pkl
import pandas as pd
from werkzeug.utils import secure_filename
import requests
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
from ar_corrector.corrector import Corrector
import time
from textblob import TextBlob
from abydos.phonetic import Soundex, Metaphone, Caverphone, NYSIIS
import pyarabic.araby as araby

db = SQLAlchemy()
app = Flask(__name__)
app.secret_key = '5accdb11b2c10a78d7c92c5fa102ea77fcd50c2058b00f6e'
app.config['UPLOAD_FOLDER'] = "static/uploads/"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///hurufToon.db"
db.init_app(app)

subscription_key_imagetotext = "7116d80f89704ae7a34493065f8c5dda"
endpoint_imagetotext = "https://huroof-toon.cognitiveservices.azure.com/"
computer_vision_client = ComputerVisionClient(
    endpoint_imagetotext, CognitiveServicesCredentials(subscription_key_imagetotext))

api_key_textcorrection = "3ff6405c5ddf4797ae9a33c95f7d276f"
endpoint_textcorrection = "https://api.bing.microsoft.com/v7.0/SpellCheck"


def image_to_text(path):
    read_image = open(path, "rb")
    read_response = computer_vision_client.read_in_stream(read_image, raw=True)
    read_operation_location = read_response.headers["Operation-Location"]
    operation_id = read_operation_location.split("/")[-1]

    while True:
        read_result = computer_vision_client.get_read_result(operation_id)
        if read_result.status.lower() not in ['notstarted', 'running']:
            break
        time.sleep(5)

    text = []
    if read_result.status == OperationStatusCodes.succeeded:
        for text_result in read_result.analyze_result.read_results:
            for line in text_result.lines:
                text.append(line.text)
    return " ".join(text)


# arabic
def spelling_text_correction_arabic(extracted_text):
    data = {'text': extracted_text}
    params = {
        'mkt': 'ar-SA',
        'mode': 'spell'
    }
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Ocp-Apim-Subscription-Key': api_key_textcorrection,
    }
    response = requests.post(endpoint_textcorrection,
                             headers=headers, params=params, data=data)
    json_response = response.json()['flaggedTokens']
    extracted_text_list = extracted_text.split(' ')
    extracted_text_list = [[extracted_text.index(extracted_text_list[i]), extracted_text_list[i]] for i in
                           range(len(extracted_text_list))]
    for correct_word in json_response:
        for false_word in extracted_text_list:
            if correct_word['offset'] == false_word[0]:
                false_word[1] = correct_word['suggestions'][0]['suggestion']
    corrected_text = " ".join([c[1] for c in extracted_text_list])
    return [corrected_text, len(json_response)]


def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1  # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def spelling_accuracy_arabic(extracted_text, corrected_text):
    return ((len(extracted_text) - (levenshtein(extracted_text, corrected_text))) / (len(extracted_text) + 1)) * 100


def percentage_of_corrections_arabic(extracted_text, number_of_mistakes):
    return number_of_mistakes / len(extracted_text.split(" ")) * 100


def gramatical_accuracy_arabic(spell_corrected):
    corr = Corrector()
    correct_text = corr.contextual_correct(spell_corrected)
    extracted_text_set = set(spell_corrected.split(" "))
    correct_text_set = set(correct_text.split(" "))
    n = max(len(extracted_text_set - correct_text_set),
            len(correct_text_set - extracted_text_set))
    return ((len(spell_corrected) - n) / (len(spell_corrected) + 1)) * 100


def get_feature_array_arabic(path: str):
    feature_array = []
    extracted_text = image_to_text(path)
    corrected_text = spelling_text_correction_arabic(extracted_text)
    feature_array.append(spelling_accuracy_arabic(extracted_text, corrected_text[0]))
    feature_array.append(gramatical_accuracy_arabic(corrected_text[0]))
    feature_array.append(percentage_of_corrections_arabic(extracted_text, corrected_text[1]))
    feature_array.append(spelling_accuracy_arabic(extracted_text, corrected_text[0]))
    return feature_array


# english
def spelling_accuracy_eng(extracted_text):
    spell_corrected = TextBlob(extracted_text).correct()
    print(spell_corrected)
    return ((len(extracted_text) - (levenshtein(extracted_text, spell_corrected))) / (len(extracted_text) + 1)) * 100


def percentage_of_corrections_eng(extracted_text):
    data = {'text': extracted_text}
    params = {
        'mkt': 'en-us',
        'mode': 'proof'
    }
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Ocp-Apim-Subscription-Key': api_key_textcorrection,
    }
    response = requests.post(endpoint_textcorrection,
                             headers=headers, params=params, data=data)
    json_response = response.json()
    return len(json_response['flaggedTokens']) / len(extracted_text.split(" ")) * 100


def percentage_of_phonetic_accuraccy_eng(extracted_text: str):
    soundex = Soundex()
    metaphone = Metaphone()
    caverphone = Caverphone()
    nysiis = NYSIIS()
    spell_corrected = TextBlob(extracted_text).correct()

    extracted_text_list = extracted_text.split(" ")
    extracted_phonetics_soundex = [soundex.encode(
        string) for string in extracted_text_list]
    extracted_phonetics_metaphone = [metaphone.encode(
        string) for string in extracted_text_list]
    extracted_phonetics_caverphone = [caverphone.encode(
        string) for string in extracted_text_list]
    extracted_phonetics_nysiis = [nysiis.encode(
        string) for string in extracted_text_list]

    extracted_soundex_string = " ".join(extracted_phonetics_soundex)
    extracted_metaphone_string = " ".join(extracted_phonetics_metaphone)
    extracted_caverphone_string = " ".join(extracted_phonetics_caverphone)
    extracted_nysiis_string = " ".join(extracted_phonetics_nysiis)

    spell_corrected_list = spell_corrected.split(" ")
    spell_corrected_phonetics_soundex = [
        soundex.encode(string) for string in spell_corrected_list]
    spell_corrected_phonetics_metaphone = [
        metaphone.encode(string) for string in spell_corrected_list]
    spell_corrected_phonetics_caverphone = [
        caverphone.encode(string) for string in spell_corrected_list]
    spell_corrected_phonetics_nysiis = [nysiis.encode(
        string) for string in spell_corrected_list]

    spell_corrected_soundex_string = " ".join(
        spell_corrected_phonetics_soundex)
    spell_corrected_metaphone_string = " ".join(
        spell_corrected_phonetics_metaphone)
    spell_corrected_caverphone_string = " ".join(
        spell_corrected_phonetics_caverphone)
    spell_corrected_nysiis_string = " ".join(spell_corrected_phonetics_nysiis)

    soundex_score = (len(extracted_soundex_string) - (levenshtein(extracted_soundex_string,
                                                                  spell_corrected_soundex_string))) / (
                            len(extracted_soundex_string) + 1)
    metaphone_score = (len(extracted_metaphone_string) - (levenshtein(extracted_metaphone_string,
                                                                      spell_corrected_metaphone_string))) / (
                              len(extracted_metaphone_string) + 1)
    caverphone_score = (len(extracted_caverphone_string) - (levenshtein(extracted_caverphone_string,
                                                                        spell_corrected_caverphone_string))) / (
                               len(extracted_caverphone_string) + 1)
    nysiis_score = (len(extracted_nysiis_string) - (levenshtein(extracted_nysiis_string,
                                                                spell_corrected_nysiis_string))) / (
                           len(extracted_nysiis_string) + 1)
    return ((0.5 * caverphone_score + 0.2 * soundex_score + 0.2 * metaphone_score + 0.1 * nysiis_score)) * 100


def get_feature_array_eng(path: str):
    feature_array = []
    extracted_text = image_to_text(path)
    feature_array.append(spelling_accuracy_eng(extracted_text))
    feature_array.append(spelling_accuracy_eng(extracted_text))
    feature_array.append(percentage_of_corrections_eng(extracted_text))
    feature_array.append(percentage_of_phonetic_accuraccy_eng(extracted_text))
    return feature_array


# français


def spelling_text_correction_french(extracted_text):
    data = {'text': extracted_text}
    params = {
        'mkt': 'fr-FR',
        'mode': 'spell'
    }
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Ocp-Apim-Subscription-Key': api_key_textcorrection,
    }
    response = requests.post(endpoint_textcorrection,
                             headers=headers, params=params, data=data)
    json_response = response.json()['flaggedTokens']
    extracted_text_list = extracted_text.split(' ')
    extracted_text_list = [[extracted_text.index(extracted_text_list[i]), extracted_text_list[i]] for i in
                           range(len(extracted_text_list))]
    for correct_word in json_response:
        for false_word in extracted_text_list:
            if correct_word['offset'] == false_word[0]:
                false_word[1] = correct_word['suggestions'][0]['suggestion']
    corrected_text = " ".join([c[1] for c in extracted_text_list])
    return [corrected_text, len(json_response)]


def get_feature_array_french(path: str):
    feature_array = []
    extracted_text = image_to_text(path)
    corrected_text = spelling_text_correction_french(extracted_text)
    feature_array.append(spelling_accuracy_arabic(extracted_text, corrected_text[0]))
    feature_array.append(spelling_accuracy_arabic(extracted_text, corrected_text[0]))
    feature_array.append(percentage_of_corrections_arabic(extracted_text, corrected_text[1]))
    feature_array.append(spelling_accuracy_arabic(extracted_text, corrected_text[0]))
    return feature_array


class User(db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String, unique=True, nullable=False)
    password = db.Column(db.String, nullable=False)
    email = db.Column(db.String, unique=True, nullable=False)
    first_name = db.Column(db.String, nullable=False)
    last_name = db.Column(db.String, nullable=False)
    birthday = db.Column(db.Date, nullable=True)
    gender = db.Column(db.String, nullable=True)
    school_year = db.Column(db.String, nullable=True)
    presence_of_dyslexia = db.Column(db.Boolean, nullable=True)
    presence_of_dysgraphia = db.Column(db.Boolean, nullable=True)
    presence_of_dyscalculia = db.Column(db.Boolean, nullable=True)
    test_taken = db.Column(db.Boolean, nullable=True)
    child = db.Column(db.Integer, nullable=True)
    feedback = db.Column(db.String, nullable=True)
    facebook_link = db.Column(db.String, nullable=True)
    instagram_link = db.Column(db.String, nullable=True)
    linkedin_link = db.Column(db.String, nullable=True)
    professional_email = db.Column(db.String, nullable=True)
    address = db.Column(db.String, nullable=True)
    role = db.Column(db.String, nullable=True)


class ExerciseAttempt(db.Model):
    __tablename__ = 'exercise_attempts'
    id = db.Column(db.Integer, primary_key=True)
    exercise_type = db.Column(db.String, nullable=False)
    exercise_level = db.Column(db.String, nullable=False)
    score = db.Column(db.Integer, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user = db.relationship('User', backref=db.backref('exercise_attempts', lazy=True))


with app.app_context():
    db.create_all()


@app.route('/')
def hello_world():
    if session:
        user = User.query.filter_by(id=session['user']).first()
        if user.role == 'STUDENT':
            if not user.test_taken:
                return redirect(url_for('test'))
        return render_template('application/index.html', user=user)
    return render_template('application/index.html', user=None)


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if session:
        return redirect(url_for('hello_world'))
    return render_template('application/signup.html')


@app.route('/signup/student', methods=['GET', 'POST'])
def signup_student():
    if session:
        return redirect(url_for('hello_world'))
    if request.method == 'POST':
        username = request.form.get('username')
        first_name = request.form.get('fname')
        last_name = request.form.get('lname')
        email = request.form.get("email")
        gender = request.form.get("gender")
        birthday = request.form.get("birthday")
        school_year = request.form.get("school_year")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")
        if username == "" or first_name == "" or last_name == "" or email == "" or gender == "" or birthday == "" or school_year == "" or password == "" or confirm_password == "":
            error = "كل المعلومات اجبارية"
            return render_template('application/signup_student.html', error=error)
        elif password != confirm_password:
            error = "كلمتا السر غير متطابقتان"
            return render_template('application/signup_student.html', error=error)
        else:
            user = User.query.filter_by(username=username).first()
            if user:
                error = "اسم الحساب غير متوفر"
                return render_template('application/signup_student.html', error=error)
        user = User(username=username, first_name=first_name, last_name=last_name, email=email, gender=gender,
                    birthday=datetime.strptime(birthday, "%Y-%m-%d").date(), school_year=school_year, password=password,
                    test_taken=False, role="STUDENT")
        db.session.add(user)
        db.session.commit()
        user = User.query.filter_by(username=username).first()
        session['user'] = user.id
        if user.role == "STUDENT":
            return redirect(url_for('test'))
        return redirect(url_for('hello_world'))
    return render_template('application/signup_student.html', error="")


@app.route('/signup/parent', methods=['GET', 'POST'])
def signup_parent():
    if session:
        return redirect(url_for('hello_world'))
    if request.method == 'POST':
        username = request.form.get('username')
        first_name = request.form.get('fname')
        last_name = request.form.get('lname')
        email = request.form.get("email")
        gender = request.form.get("gender")
        birthday = request.form.get("birthday")
        child_username = request.form.get("child")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")
        if username == "" or first_name == "" or last_name == "" or email == "" or gender == "" or birthday == "" or child_username == "" or password == "" or confirm_password == "":
            error = "كل المعلومات اجبارية"
            return render_template('application/signup_parent.html', error=error)
        elif password != confirm_password:
            error = "كلمتا السر غير متطابقتان"
            return render_template('application/signup_parent.html', error=error)
        else:
            user = User.query.filter_by(username=username).first()
            if user:
                error = "اسم الحساب غير متوفر"
                return render_template('application/signup_parent.html', error=error)
            child = User.query.filter_by(username=child_username).first()
            if not child:
                error = "لا يوجد حساب طفل بهذا الاسم"
                return render_template('application/signup_parent.html', error=error)
            else:
                child = User.query.filter_by(username=child_username).first()
                child_username = child.id
        user = User(username=username, first_name=first_name, last_name=last_name, email=email, gender=gender,
                    birthday=datetime.strptime(birthday, "%Y-%m-%d").date(), child=child_username, password=password,
                    role="PARENT")
        db.session.add(user)
        db.session.commit()
        user = User.query.filter_by(username=username).first()
        session['user'] = user.id
        return redirect(url_for('hello_world'))
    return render_template('application/signup_parent.html', error="")


@app.route('/about', methods=['GET', 'POST'])
def about():
    if session:
        user = User.query.filter_by(id=session['user']).first()
        if user.role == "STUDENT" and not user.test_taken:
            return redirect(url_for('test'))
        return render_template('application/about.html', user=user)
    return render_template('application/about.html')


@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if session:
        user = User.query.filter_by(id=session['user']).first()
        if user.role == "STUDENT" and not user.test_taken:
            return redirect(url_for('test'))
        return render_template('application/contact.html', user=user)
    return render_template('application/contact.html')


@app.route('/signup/speech-therapist', methods=['GET', 'POST'])
def signup_therapist():
    if session:
        return redirect(url_for('hello_world'))
    if request.method == 'POST':
        username = request.form.get('username')
        first_name = request.form.get('fname')
        last_name = request.form.get('lname')
        email = request.form.get("email")
        gender = request.form.get("gender")
        birthday = request.form.get("birthday")
        feedback = request.form.get("feedback")
        facebook = request.form.get("facebook")
        instagram = request.form.get("instagram")
        linkedin = request.form.get("linkedin")
        address = request.form.get("address")
        professional_email = request.form.get("email_pro")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")
        if username == "" or first_name == "" or last_name == "" or email == "" or gender == "" or birthday == "" or facebook == "" or linkedin == "" or instagram == "" or professional_email == "" or feedback == "" or address == "" or password == "" or confirm_password == "":
            error = "كل المعلومات اجبارية"
            return render_template('application/signup_therapist.html', error=error)
        elif password != confirm_password:
            error = "كلمتا السر غير متطابقتان"
            return render_template('application/signup_therapist.html', error=error)
        else:
            user = User.query.filter_by(username=username).first()
            if user:
                error = "اسم الحساب غير متوفر"
                return render_template('application/signup_therapist.html', error=error)
        user = User(username=username, first_name=first_name, last_name=last_name, email=email, gender=gender,
                    birthday=datetime.strptime(birthday, "%Y-%m-%d").date(), facebook_link=facebook,
                    instagram_link=instagram, linkedin_link=linkedin, feedback=feedback, address=address,
                    password=password, professional_email=professional_email, role="THERAPIST")
        db.session.add(user)
        db.session.commit()
        user = User.query.filter_by(username=username).first()
        session['user'] = user.id
        return redirect(url_for('hello_world'))
    return render_template('application/signup_therapist.html')


@app.route('/logout', methods=['GET', 'POST'])
def logout():
    session.clear()
    return redirect(url_for('hello_world'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if session:
        return redirect(url_for('hello_world'))
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form.get('username')).first()
        if not user:
            error = 'لا يوجد حساب بهذا الاسم'
            return render_template('application/login.html', error=error)
        if user.password != request.form.get('password'):
            error = 'كلمة السر خاطئة'
            return render_template('application/login.html', error=error)
        session['user'] = user.id
        return redirect(url_for('hello_world'))
    return render_template('application/login.html', error="")


@app.route('/delete', methods=['GET', 'POST'])
def delete_account():
    if not session:
        return redirect(url_for('hello_world'))
    User.query.filter_by(id=session['user']).delete()
    session.clear()
    db.session.commit()
    return render_template('application/index.html')


@app.route('/edit-account', methods=['GET', 'POST'])
def edit_account():
    if not session:
        return redirect(url_for('hello_world'))
    user = User.query.filter_by(id=session['user']).first()
    child_username_disp = ""
    if user.role == "PARENT":
        child_username_disp = User.query.filter_by(id=user.child).first().username
    if request.method == 'POST':
        if user.role == "STUDENT":
            username = request.form.get('username')
            first_name = request.form.get('fname')
            last_name = request.form.get('lname')
            email = request.form.get("email")
            gender = request.form.get("gender")
            birthday = request.form.get("birthday")
            school_year = request.form.get("school_year")
            password = request.form.get("password")
            confirm_password = request.form.get("confirm_password")
            if username == "" or first_name == "" or last_name == "" or email == "" or gender == "" or birthday == "" or school_year == "" or password == "" or confirm_password == "":
                error = "كل المعلومات اجبارية"
                return render_template('application/edit-account.html', error=error, user=user)
            elif password != confirm_password:
                error = "كلمتا السر غير متطابقتان"
                return render_template('application/edit-account.html', error=error, user=user)
            else:
                user2 = User.query.filter_by(username=username).first()
                if user2:
                    if user2.id != user.id:
                        error = "اسم الحساب غير متوفر"
                        return render_template('application/edit-account.html', error=error, user=user)
            user.username = username
            user.first_name = first_name
            user.last_name = last_name
            user.email = email
            user.gender = gender
            user.birthday = datetime.strptime(birthday, "%Y-%m-%d").date()
            user.school_year = school_year
            user.password = password
        elif user.role == "THERAPIST":
            username = request.form.get('username')
            first_name = request.form.get('fname')
            last_name = request.form.get('lname')
            email = request.form.get("email")
            gender = request.form.get("gender")
            birthday = request.form.get("birthday")
            feedback = request.form.get("feedback")
            facebook = request.form.get("facebook")
            instagram = request.form.get("instagram")
            linkedin = request.form.get("linkedin")
            address = request.form.get("address")
            professional_email = request.form.get("email_pro")
            password = request.form.get("password")
            confirm_password = request.form.get("confirm_password")
            if username == "" or first_name == "" or last_name == "" or email == "" or gender == "" or birthday == "" or facebook == "" or linkedin == "" or instagram == "" or professional_email == "" or feedback == "" or address == "" or password == "" or confirm_password == "":
                error = "كل المعلومات اجبارية"
                return render_template('application/edit-account.html', error=error, user=user)
            elif password != confirm_password:
                error = "كلمتا السر غير متطابقتان"
                return render_template('application/edit-account.html', error=error, user=user)
            else:
                user2 = User.query.filter_by(username=username).first()
                if user2:
                    if user2.id != user.id:
                        error = "اسم الحساب غير متوفر"
                        return render_template('application/edit-account.html', error=error, user=user)
            user.username = username
            user.first_name = first_name
            user.last_name = last_name
            user.email = email
            user.gender = gender
            user.birthday = datetime.strptime(birthday, "%Y-%m-%d").date()
            user.facebook_link = facebook
            user.instagram_link = instagram
            user.linkedin_link = linkedin
            user.professional_email = professional_email
            user.address = address
            user.feedback = feedback
            user.password = password
        elif user.role == "PARENT":
            username = request.form.get('username')
            first_name = request.form.get('fname')
            last_name = request.form.get('lname')
            email = request.form.get("email")
            gender = request.form.get("gender")
            birthday = request.form.get("birthday")
            child_username = request.form.get("child")
            password = request.form.get("password")
            confirm_password = request.form.get("confirm_password")
            if username == "" or first_name == "" or last_name == "" or email == "" or gender == "" or birthday == "" or child_username == "" or password == "" or confirm_password == "":
                error = "كل المعلومات اجبارية"
                return render_template('application/edit-account.html', error=error, user=user)
            elif password != confirm_password:
                error = "كلمتا السر غير متطابقتان"
                return render_template('application/edit-account.html', error=error, user=user)
            else:
                user2 = User.query.filter_by(username=username).first()
                if user2:
                    if user2.id != user.id:
                        error = "اسم الحساب غير متوفر"
                        return render_template('application/edit-account.html', error=error, user=user)
                child = User.query.filter_by(username=child_username).first()
                if not child:
                    error = "لا يوجد حساب طفل بهذا الاسم"
                    return render_template('application/edit-account.html', error=error, user=user)
                else:
                    child_username = child.id
            user.username = username
            user.first_name = first_name
            user.last_name = last_name
            user.email = email
            user.gender = gender
            user.birthday = datetime.strptime(birthday, "%Y-%m-%d").date()
            user.child = child_username
        db.session.add(user)
        db.session.commit()
    return render_template('application/edit-account.html', user=user, child_username=child_username_disp)


@app.route('/speech-therapist/all', methods=['GET', 'POST'])
def get_therapists():
    if not session:
        return redirect(url_for('hello_world'))
    user = User.query.filter_by(id=session['user']).first()
    if user.role != "PARENT":
        return redirect(url_for('hello_world'))
    therapists = User.query.filter_by(role="THERAPIST")
    return render_template('application/therapists.html', therapists=therapists, user=user)


@app.route('/test', methods=['GET', 'POST'])
def test():
    if not session:
        return redirect(url_for('hello_world'))
    user = User.query.filter_by(id=session['user']).first()
    if user.role != "STUDENT":
        return redirect(url_for('hello_world'))
    if user.test_taken:
        return redirect(url_for('hello_world'))
    return render_template('application/test.html', user=user)


@app.route('/test/arabic', methods=['GET', 'POST'])
def test_arabic():
    if not session:
        return redirect(url_for('hello_world'))
    user = User.query.filter_by(id=session['user']).first()
    if user.role != "STUDENT":
        return redirect(url_for('hello_world'))
    if user.test_taken:
        return redirect(url_for('hello_world'))
    if request.method == 'POST':
        q1 = request.form.get('q1')
        q2 = request.form.get('q2')
        q3 = request.form.get('q3')
        q4 = request.form.get('q4')
        q5 = request.form.get('q5')
        q6 = request.form.get('q6')
        q71 = request.form.get('q71')
        q72 = request.form.get('q72')
        q73 = request.form.get('q73')
        q74 = request.form.get('q74')
        q75 = request.form.get('q75')
        q76 = request.form.get('q76')
        q9 = request.form.get('q9')
        dyscalculia_array = [int(q76), int(q75), int(q74), int(q73), int(q72), int(q71)]
        comparing_score = 0
        for i in range(len(dyscalculia_array)):
            for j in range(i + 1, len(dyscalculia_array)):
                if dyscalculia_array[i] < dyscalculia_array[j]:
                    comparing_score += 1
        doubling_score = abs(int(q1) - 16)
        counting_score = abs(int(q2) - 13)
        writing_score = abs(int(q3) - 96)
        symbol_a = 1 if q4 == 'a' else 0
        symbol_b = 1 if q4 == 'b' else 0
        symbol_c = 1 if q4 == 'c' else 0
        symbol_d = 1 if q4 == 'd' else 0
        addition_score = abs(int(q5) - 26)
        multiplication_score = abs(int(q6) - 21)
        user_age = int((datetime.now().date() - user.birthday).days / 362.25)
        user_female = 1 if user.gender == 'FEMALE' else 0
        user_male = 1 if user.gender == 'MALE' else 0
        final_dyscalculia_array = pd.DataFrame([[user_age, doubling_score, counting_score, writing_score,
                                                 addition_score, multiplication_score, comparing_score, user_female,
                                                 user_male, symbol_a, symbol_b, symbol_c, symbol_d, 0]],
                                               columns=["Age", "Doubling Numbers", "Counting", "Writing Numbers",
                                                        "Addition", "Multiplication", "Comparing Numbers",
                                                        "Gender_female", "Gender_male", "Symbol Recognition_a",
                                                        "Symbol Recognition_b", "Symbol Recognition_c",
                                                        "Symbol Recognition_d", "Symbol Recognition_e"])
        has_dyscalculia = False
        has_dysgraphia = False
        has_dyslexia = False
        with open("models/dyscal_model.sav", 'rb') as file:
            dyscalculia_model = pkl.load(file)
            has_dyscalculia = True if dyscalculia_model.predict(final_dyscalculia_array)[0] == 1 else False
        file = request.files['q8']
        filename = f"{user.username}_dysgraphia_file." + secure_filename(file.filename).split(".")[-1]
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        feature_array = pd.DataFrame([get_feature_array_arabic(os.path.join(app.config['UPLOAD_FOLDER'], filename))],
                                     columns=["spelling_accuracy", "gramatical_accuracy", " percentage_of_corrections",
                                              "percentage_of_phonetic_accuraccy"])
        with open("models/dysgraphia_model.sav", 'rb') as file:
            dysgraphia_model = pkl.load(file)
            res = dysgraphia_model.predict(feature_array)
            has_dysgraphia = True if res[0] == 1 else False
        correct_string = "إذا الشعب يوما أراد الحياة، فلا بد أن يستجيب القدر."
        similarity_score = 100 * (1 - levenshtein(correct_string, q9) / len(correct_string))
        has_dyslexia = True if similarity_score < 70 else False
        user.test_taken = True
        user.presence_of_dyslexia = has_dyslexia
        user.presence_of_dysgraphia = has_dysgraphia
        user.presence_of_dyscalculia = has_dyscalculia
        db.session.add(user)
        db.session.commit()
        return redirect(url_for('test_result'))
    return render_template('application/test_arabic.html', user=user)


@app.route('/test/french', methods=['GET', 'POST'])
def test_french():
    if not session:
        return redirect(url_for('hello_world'))
    user = User.query.filter_by(id=session['user']).first()
    if user.role != "STUDENT":
        return redirect(url_for('hello_world'))
    if user.test_taken:
        return redirect(url_for('hello_world'))
    if request.method == 'POST':
        q1 = request.form.get('q1')
        q2 = request.form.get('q2')
        q3 = request.form.get('q3')
        q4 = request.form.get('q4')
        q5 = request.form.get('q5')
        q6 = request.form.get('q6')
        q71 = request.form.get('q71')
        q72 = request.form.get('q72')
        q73 = request.form.get('q73')
        q74 = request.form.get('q74')
        q75 = request.form.get('q75')
        q76 = request.form.get('q76')
        q9 = request.form.get('q9')
        dyscalculia_array = [int(q76), int(q75), int(q74), int(q73), int(q72), int(q71)]
        comparing_score = 0
        for i in range(len(dyscalculia_array)):
            for j in range(i + 1, len(dyscalculia_array)):
                if dyscalculia_array[i] < dyscalculia_array[j]:
                    comparing_score += 1
        doubling_score = abs(int(q1) - 16)
        counting_score = abs(int(q2) - 13)
        writing_score = abs(int(q3) - 96)
        symbol_a = 1 if q4 == 'a' else 0
        symbol_b = 1 if q4 == 'b' else 0
        symbol_c = 1 if q4 == 'c' else 0
        symbol_d = 1 if q4 == 'd' else 0
        addition_score = abs(int(q5) - 26)
        multiplication_score = abs(int(q6) - 21)
        user_age = int((datetime.now().date() - user.birthday).days / 362.25)
        user_female = 1 if user.gender == 'FEMALE' else 0
        user_male = 1 if user.gender == 'MALE' else 0
        final_dyscalculia_array = pd.DataFrame([[user_age, doubling_score, counting_score, writing_score,
                                                 addition_score, multiplication_score, comparing_score, user_female,
                                                 user_male, symbol_a, symbol_b, symbol_c, symbol_d, 0]],
                                               columns=["Age", "Doubling Numbers", "Counting", "Writing Numbers",
                                                        "Addition", "Multiplication", "Comparing Numbers",
                                                        "Gender_female", "Gender_male", "Symbol Recognition_a",
                                                        "Symbol Recognition_b", "Symbol Recognition_c",
                                                        "Symbol Recognition_d", "Symbol Recognition_e"])
        has_dyscalculia = False
        has_dysgraphia = False
        has_dyslexia = False
        with open("models/dyscal_model.sav", 'rb') as file:
            dyscalculia_model = pkl.load(file)
            has_dyscalculia = True if dyscalculia_model.predict(final_dyscalculia_array)[0] == 1 else False
        file = request.files['q8']
        filename = f"{user.username}_dysgraphia_file." + secure_filename(file.filename).split(".")[-1]
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        feature_array = pd.DataFrame([get_feature_array_french(os.path.join(app.config['UPLOAD_FOLDER'], filename))],
                                     columns=["spelling_accuracy", "gramatical_accuracy", " percentage_of_corrections",
                                              "percentage_of_phonetic_accuraccy"])
        with open("models/dysgraphia_model.sav", 'rb') as file:
            dysgraphia_model = pkl.load(file)
            res = dysgraphia_model.predict(feature_array)
            has_dysgraphia = True if res[0] == 1 else False
        correct_string = "Il faut travailler beaucoup pour réussir."
        similarity_score = 100 * (1 - levenshtein(correct_string, q9) / len(correct_string))
        has_dyslexia = True if similarity_score < 70 else False
        user.test_taken = True
        user.presence_of_dyslexia = has_dyslexia
        user.presence_of_dysgraphia = has_dysgraphia
        user.presence_of_dyscalculia = has_dyscalculia
        db.session.add(user)
        db.session.commit()
        return redirect(url_for('test_result'))
    return render_template('application/test_french.html')


@app.route('/test/english', methods=['GET', 'POST'])
def test_english():
    if not session:
        return redirect(url_for('hello_world'))
    user = User.query.filter_by(id=session['user']).first()
    if user.role != "STUDENT":
        return redirect(url_for('hello_world'))
    if user.test_taken:
        return redirect(url_for('hello_world'))
    if request.method == 'POST':
        q1 = request.form.get('q1')
        q2 = request.form.get('q2')
        q3 = request.form.get('q3')
        q4 = request.form.get('q4')
        q5 = request.form.get('q5')
        q6 = request.form.get('q6')
        q71 = request.form.get('q71')
        q72 = request.form.get('q72')
        q73 = request.form.get('q73')
        q74 = request.form.get('q74')
        q75 = request.form.get('q75')
        q76 = request.form.get('q76')
        q9 = request.form.get('q9')
        dyscalculia_array = [int(q76), int(q75), int(q74), int(q73), int(q72), int(q71)]
        comparing_score = 0
        for i in range(len(dyscalculia_array)):
            for j in range(i + 1, len(dyscalculia_array)):
                if dyscalculia_array[i] < dyscalculia_array[j]:
                    comparing_score += 1
        doubling_score = abs(int(q1) - 16)
        counting_score = abs(int(q2) - 13)
        writing_score = abs(int(q3) - 96)
        symbol_a = 1 if q4 == 'a' else 0
        symbol_b = 1 if q4 == 'b' else 0
        symbol_c = 1 if q4 == 'c' else 0
        symbol_d = 1 if q4 == 'd' else 0
        addition_score = abs(int(q5) - 26)
        multiplication_score = abs(int(q6) - 21)
        user_age = int((datetime.now().date() - user.birthday).days / 362.25)
        user_female = 1 if user.gender == 'FEMALE' else 0
        user_male = 1 if user.gender == 'MALE' else 0
        final_dyscalculia_array = pd.DataFrame([[user_age, doubling_score, counting_score, writing_score,
                                                 addition_score, multiplication_score, comparing_score, user_female,
                                                 user_male, symbol_a, symbol_b, symbol_c, symbol_d, 0]],
                                               columns=["Age", "Doubling Numbers", "Counting", "Writing Numbers",
                                                        "Addition", "Multiplication", "Comparing Numbers",
                                                        "Gender_female", "Gender_male", "Symbol Recognition_a",
                                                        "Symbol Recognition_b", "Symbol Recognition_c",
                                                        "Symbol Recognition_d", "Symbol Recognition_e"])
        has_dyscalculia = False
        has_dysgraphia = False
        has_dyslexia = False
        with open("models/dyscal_model.sav", 'rb') as file:
            dyscalculia_model = pkl.load(file)
            has_dyscalculia = True if dyscalculia_model.predict(final_dyscalculia_array)[0] == 1 else False
        file = request.files['q8']
        filename = f"{user.username}_dysgraphia_file." + secure_filename(file.filename).split(".")[-1]
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        feature_array = pd.DataFrame([get_feature_array_eng(os.path.join(app.config['UPLOAD_FOLDER'], filename))],
                                     columns=["spelling_accuracy", "gramatical_accuracy", " percentage_of_corrections",
                                              "percentage_of_phonetic_accuraccy"])
        with open("models/dysgraphia_model.sav", 'rb') as file:
            dysgraphia_model = pkl.load(file)
            res = dysgraphia_model.predict(feature_array)
            has_dysgraphia = True if res[0] == 1 else False
        correct_string = "I will go to school tomorrow."
        similarity_score = 100 * (1 - levenshtein(correct_string, q9) / len(correct_string))
        has_dyslexia = True if similarity_score < 70 else False
        user.test_taken = True
        user.presence_of_dyslexia = has_dyslexia
        user.presence_of_dysgraphia = has_dysgraphia
        user.presence_of_dyscalculia = has_dyscalculia
        db.session.add(user)
        db.session.commit()
        return redirect(url_for('test_result'))
    return render_template('application/test_english.html', user=user)


@app.route('/test/result', methods=['GET', 'POST'])
def test_result():
    if not session:
        return redirect(url_for('hello_world'))
    user = User.query.filter_by(id=session['user']).first()
    if user.role != "STUDENT":
        return redirect(url_for('hello_world'))
    if not user.test_taken:
        return redirect(url_for('hello_world'))
    return render_template('application/test_result.html', x=user.presence_of_dyslexia, g=user.presence_of_dysgraphia,
                           c=user.presence_of_dyscalculia, user=user)


@app.route('/play', methods=['GET', 'POST'])
def play():
    if not session:
        return redirect(url_for('hello_world'))
    else:
        user = User.query.filter_by(id=session['user']).first()
        if user.role != "STUDENT":
            return redirect(url_for('hello_world'))
        if user.role == "STUDENT" and not user.test_taken:
            return redirect(url_for('test'))
        return render_template('application/play.html', user=user)


@app.route('/arabic/dictee/easy', methods=['GET', 'POST'])
def arabic_dictee():
    if not session:
        return redirect(url_for('hello_world'))
    else:
        user = User.query.filter_by(id=session['user']).first()
        if user.role != "STUDENT":
            return redirect(url_for('hello_world'))
        if user.role == "STUDENT" and not user.test_taken:
            return redirect(url_for('test'))
        if user.role == "STUDENT" and user.school_year not in ["1", "2"]:
            if user.school_year in ["3", "4"]:
                return redirect(url_for('arabic_dictee_medium'))
            if user.school_year in ["5", "6"]:
                return redirect(url_for('arabic_dictee_hard'))
    words_list = []
    with open("level-1-arabic.txt", "r", encoding="utf-8") as file:
        text = file.read()
        words_list = text.split('\n')
    exercise_list = random.sample(words_list, 3)
    if request.method == "POST":
        answer1 = araby.strip_diacritics(request.form.get('q1'))
        answer2 = araby.strip_diacritics(request.form.get('q2'))
        answer3 = araby.strip_diacritics(request.form.get('q3'))
        correct_answer1 = araby.strip_diacritics(request.form.get('a1'))
        correct_answer2 = araby.strip_diacritics(request.form.get('a2'))
        correct_answer3 = araby.strip_diacritics(request.form.get('a3'))
        score1 = 100 * (1 - levenshtein(correct_answer1, answer1) / len(correct_answer1))
        score2 = 100 * (1 - levenshtein(correct_answer2, answer2) / len(correct_answer2))
        score3 = 100 * (1 - levenshtein(correct_answer3, answer3) / len(correct_answer3))
        final_score = int((score1 + score2 + score3) / 15)
        exercise_attempt = ExerciseAttempt(user_id=session.get("user"), score=final_score, exercise_type="ARAB_DICTEE",
                                           exercise_level="EASY")
        db.session.add(exercise_attempt)
        db.session.commit()
        return render_template('application/result_exercise.html', score=final_score, user=user)
    return render_template('application/arabic_dictee.html', user=user, words=exercise_list)


@app.route('/arabic/dictee/medium', methods=['GET', 'POST'])
def arabic_dictee_medium():
    if not session:
        return redirect(url_for('hello_world'))
    else:
        user = User.query.filter_by(id=session['user']).first()
        if user.role != "STUDENT":
            return redirect(url_for('hello_world'))
        if user.role == "STUDENT" and not user.test_taken:
            return redirect(url_for('test'))
        if user.role == "STUDENT" and user.school_year not in ["3", "4"]:
            if user.school_year in ["1", "2"]:
                return redirect(url_for('arabic_dictee'))
            if user.school_year in ["5", "6"]:
                return redirect(url_for('arabic_dictee_hard'))
    words_list = []
    with open("level-2-arabic.txt", "r", encoding="utf-8") as file:
        text = file.read()
        words_list = text.split('\n')
    exercise_list = random.sample(words_list, 3)
    if request.method == "POST":
        answer1 = araby.strip_diacritics(request.form.get('q1'))
        answer2 = araby.strip_diacritics(request.form.get('q2'))
        answer3 = araby.strip_diacritics(request.form.get('q3'))
        correct_answer1 = araby.strip_diacritics(request.form.get('a1'))
        correct_answer2 = araby.strip_diacritics(request.form.get('a2'))
        correct_answer3 = araby.strip_diacritics(request.form.get('a3'))
        score1 = 100 * (1 - levenshtein(correct_answer1, answer1) / len(correct_answer1))
        score2 = 100 * (1 - levenshtein(correct_answer2, answer2) / len(correct_answer2))
        score3 = 100 * (1 - levenshtein(correct_answer3, answer3) / len(correct_answer3))
        final_score = int((score1 + score2 + score3) / 15)
        exercise_attempt = ExerciseAttempt(user_id=session.get("user"), score=final_score, exercise_type="ARAB_DICTEE",
                                           exercise_level="MEDIUM")
        db.session.add(exercise_attempt)
        db.session.commit()
        return render_template('application/result_exercise.html', score=final_score, user=user)
    return render_template('application/arabic_dictee.html', user=user, words=exercise_list)


@app.route('/arabic/dictee/hard', methods=['GET', 'POST'])
def arabic_dictee_hard():
    if not session:
        return redirect(url_for('hello_world'))
    else:
        user = User.query.filter_by(id=session['user']).first()
        if user.role != "STUDENT":
            return redirect(url_for('hello_world'))
        if user.role == "STUDENT" and not user.test_taken:
            return redirect(url_for('test'))
        if user.role == "STUDENT" and user.school_year not in ["5", "6"]:
            print("redirection 3")
            if user.school_year in ["1", "2"]:
                return redirect(url_for('arabic_dictee'))
            if user.school_year in ["3", "4"]:
                return redirect(url_for('arabic_dictee_medium'))
    words_list = []
    with open("level-3-arabic.txt", "r", encoding="utf-8") as file:
        text = file.read()
        words_list = text.split('\n')
    exercise_list = random.sample(words_list, 3)
    if request.method == "POST":
        answer1 = araby.strip_diacritics(request.form.get('q1'))
        answer2 = araby.strip_diacritics(request.form.get('q2'))
        answer3 = araby.strip_diacritics(request.form.get('q3'))
        correct_answer1 = araby.strip_diacritics(request.form.get('a1'))
        correct_answer2 = araby.strip_diacritics(request.form.get('a2'))
        correct_answer3 = araby.strip_diacritics(request.form.get('a3'))
        score1 = 100 * (1 - levenshtein(correct_answer1, answer1) / len(correct_answer1))
        score2 = 100 * (1 - levenshtein(correct_answer2, answer2) / len(correct_answer2))
        score3 = 100 * (1 - levenshtein(correct_answer3, answer3) / len(correct_answer3))
        final_score = int((score1 + score2 + score3) / 15)
        exercise_attempt = ExerciseAttempt(user_id=session.get("user"), score=final_score, exercise_type="ARAB_DICTEE",
                                           exercise_level="HARD")
        db.session.add(exercise_attempt)
        db.session.commit()
        return render_template('application/result_exercise.html', score=final_score, user=user)
    return render_template('application/arabic_dictee.html', user=user, words=exercise_list)


@app.route('/english/dictee/easy', methods=['GET', 'POST'])
def english_dictee():
    if not session:
        return redirect(url_for('hello_world'))
    else:
        user = User.query.filter_by(id=session['user']).first()
        if user.role != "STUDENT":
            return redirect(url_for('hello_world'))
        if user.role == "STUDENT" and not user.test_taken:
            return redirect(url_for('test'))
        if user.role == "STUDENT" and user.school_year not in ["1", "2"]:
            if user.school_year in ["3", "4"]:
                return redirect(url_for('english_dictee_medium'))
            if user.school_year in ["5", "6"]:
                return redirect(url_for('english_dictee_hard'))
    words_list = []
    with open("level-1-english.txt", "r", encoding="utf-8") as file:
        text = file.read()
        words_list = text.split('\n')
    exercise_list = random.sample(words_list, 3)
    if request.method == "POST":
        answer1 = request.form.get('q1')
        answer2 = request.form.get('q2')
        answer3 = request.form.get('q3')
        correct_answer1 = request.form.get('a1')
        correct_answer2 = request.form.get('a2')
        correct_answer3 = request.form.get('a3')
        score1 = 100 * (1 - levenshtein(correct_answer1, answer1) / len(correct_answer1))
        score2 = 100 * (1 - levenshtein(correct_answer2, answer2) / len(correct_answer2))
        score3 = 100 * (1 - levenshtein(correct_answer3, answer3) / len(correct_answer3))
        final_score = int((score1 + score2 + score3) / 15)
        exercise_attempt = ExerciseAttempt(user_id=session.get("user"), score=final_score, exercise_type="ENGLISH_DICTEE",
                                           exercise_level="EASY")
        db.session.add(exercise_attempt)
        db.session.commit()
        return render_template('application/result_exercise.html', score=final_score, user=user)
    return render_template('application/english_dictee.html', user=user, words=exercise_list)


@app.route('/english/dictee/medium', methods=['GET', 'POST'])
def english_dictee_medium():
    if not session:
        return redirect(url_for('hello_world'))
    else:
        user = User.query.filter_by(id=session['user']).first()
        if user.role != "STUDENT":
            return redirect(url_for('hello_world'))
        if user.role == "STUDENT" and not user.test_taken:
            return redirect(url_for('test'))
        if user.role == "STUDENT" and user.school_year not in ["3", "4"]:
            if user.school_year in ["1", "2"]:
                return redirect(url_for('english_dictee'))
            if user.school_year in ["5", "6"]:
                return redirect(url_for('english_dictee_hard'))
    words_list = []
    with open("level-2-english.txt", "r", encoding="utf-8") as file:
        text = file.read()
        words_list = text.split('\n')
    exercise_list = random.sample(words_list, 3)
    if request.method == "POST":
        answer1 = request.form.get('q1')
        answer2 = request.form.get('q2')
        answer3 = request.form.get('q3')
        correct_answer1 = request.form.get('a1')
        correct_answer2 = request.form.get('a2')
        correct_answer3 = request.form.get('a3')
        score1 = 100 * (1 - levenshtein(correct_answer1, answer1) / len(correct_answer1))
        score2 = 100 * (1 - levenshtein(correct_answer2, answer2) / len(correct_answer2))
        score3 = 100 * (1 - levenshtein(correct_answer3, answer3) / len(correct_answer3))
        final_score = int((score1 + score2 + score3) / 15)
        exercise_attempt = ExerciseAttempt(user_id=session.get("user"), score=final_score, exercise_type="ENGLISH_DICTEE",
                                           exercise_level="MEDIUM")
        db.session.add(exercise_attempt)
        db.session.commit()
        return render_template('application/result_exercise.html', score=final_score, user=user)
    return render_template('application/english_dictee.html', user=user, words=exercise_list)


@app.route('/english/dictee/hard', methods=['GET', 'POST'])
def english_dictee_hard():
    if not session:
        return redirect(url_for('hello_world'))
    else:
        user = User.query.filter_by(id=session['user']).first()
        if user.role != "STUDENT":
            return redirect(url_for('hello_world'))
        if user.role == "STUDENT" and not user.test_taken:
            return redirect(url_for('test'))
        if user.role == "STUDENT" and user.school_year not in ["5", "6"]:
            if user.school_year in ["1", "2"]:
                return redirect(url_for('english_dictee'))
            if user.school_year in ["3", "4"]:
                return redirect(url_for('english_dictee_medium'))
    words_list = []
    with open("level-3-english.txt", "r", encoding="utf-8") as file:
        text = file.read()
        words_list = text.split('\n')
    exercise_list = random.sample(words_list, 3)
    if request.method == "POST":
        answer1 = request.form.get('q1')
        answer2 = request.form.get('q2')
        answer3 = request.form.get('q3')
        correct_answer1 = request.form.get('a1')
        correct_answer2 = request.form.get('a2')
        correct_answer3 = request.form.get('a3')
        score1 = 100 * (1 - levenshtein(correct_answer1, answer1) / len(correct_answer1))
        score2 = 100 * (1 - levenshtein(correct_answer2, answer2) / len(correct_answer2))
        score3 = 100 * (1 - levenshtein(correct_answer3, answer3) / len(correct_answer3))
        final_score = int((score1 + score2 + score3) / 15)
        exercise_attempt = ExerciseAttempt(user_id=session.get("user"), score=final_score, exercise_type="ENGLISH_DICTEE",
                                           exercise_level="HARD")
        db.session.add(exercise_attempt)
        db.session.commit()
        return render_template('application/result_exercise.html', score=final_score, user=user)
    return render_template('application/english_dictee.html', user=user, words=exercise_list)


@app.route('/french/dictee/easy', methods=['GET', 'POST'])
def french_dictee():
    if not session:
        return redirect(url_for('hello_world'))
    else:
        user = User.query.filter_by(id=session['user']).first()
        if user.role != "STUDENT":
            return redirect(url_for('hello_world'))
        if user.role == "STUDENT" and not user.test_taken:
            return redirect(url_for('test'))
        if user.role == "STUDENT" and user.school_year not in ["1", "2"]:
            if user.school_year in ["3", "4"]:
                return redirect(url_for('french_dictee_medium'))
            if user.school_year in ["5", "6"]:
                return redirect(url_for('french_dictee_hard'))
    words_list = []
    with open("level-1-french.txt", "r", encoding="utf-8") as file:
        text = file.read()
        words_list = text.split('\n')
    exercise_list = random.sample(words_list, 3)
    if request.method == "POST":
        answer1 = request.form.get('q1')
        answer2 = request.form.get('q2')
        answer3 = request.form.get('q3')
        correct_answer1 = request.form.get('a1')
        correct_answer2 = request.form.get('a2')
        correct_answer3 = request.form.get('a3')
        score1 = 100 * (1 - levenshtein(correct_answer1, answer1) / len(correct_answer1))
        score2 = 100 * (1 - levenshtein(correct_answer2, answer2) / len(correct_answer2))
        score3 = 100 * (1 - levenshtein(correct_answer3, answer3) / len(correct_answer3))
        final_score = int((score1 + score2 + score3) / 15)
        exercise_attempt = ExerciseAttempt(user_id=session.get("user"), score=final_score, exercise_type="FRENCH_DICTEE",
                                           exercise_level="EASY")
        db.session.add(exercise_attempt)
        db.session.commit()
        return render_template('application/result_exercise.html', score=final_score, user=user)
    return render_template('application/french_dictee.html', user=user, words=exercise_list)


@app.route('/french/dictee/medium', methods=['GET', 'POST'])
def french_dictee_medium():
    if not session:
        return redirect(url_for('hello_world'))
    else:
        user = User.query.filter_by(id=session['user']).first()
        if user.role != "STUDENT":
            return redirect(url_for('hello_world'))
        if user.role == "STUDENT" and not user.test_taken:
            return redirect(url_for('test'))
        if user.role == "STUDENT" and user.school_year not in ["3", "4"]:
            if user.school_year in ["1", "2"]:
                return redirect(url_for('french_dictee'))
            if user.school_year in ["5", "6"]:
                return redirect(url_for('french_dictee_hard'))
    words_list = []
    with open("level-2-french.txt", "r", encoding="utf-8") as file:
        text = file.read()
        words_list = text.split('\n')
    exercise_list = random.sample(words_list, 3)
    if request.method == "POST":
        answer1 = request.form.get('q1')
        answer2 = request.form.get('q2')
        answer3 = request.form.get('q3')
        correct_answer1 = request.form.get('a1')
        correct_answer2 = request.form.get('a2')
        correct_answer3 = request.form.get('a3')
        score1 = 100 * (1 - levenshtein(correct_answer1, answer1) / len(correct_answer1))
        score2 = 100 * (1 - levenshtein(correct_answer2, answer2) / len(correct_answer2))
        score3 = 100 * (1 - levenshtein(correct_answer3, answer3) / len(correct_answer3))
        final_score = int((score1 + score2 + score3) / 15)
        exercise_attempt = ExerciseAttempt(user_id=session.get("user"), score=final_score, exercise_type="FRENCH_DICTEE",
                                           exercise_level="MEDIUM")
        db.session.add(exercise_attempt)
        db.session.commit()
        return render_template('application/result_exercise.html', score=final_score, user=user)
    return render_template('application/french_dictee.html', user=user, words=exercise_list)


@app.route('/french/dictee/hard', methods=['GET', 'POST'])
def french_dictee_hard():
    if not session:
        return redirect(url_for('hello_world'))
    else:
        user = User.query.filter_by(id=session['user']).first()
        if user.role != "STUDENT":
            return redirect(url_for('hello_world'))
        if user.role == "STUDENT" and not user.test_taken:
            return redirect(url_for('test'))
        if user.role == "STUDENT" and user.school_year not in ["5", "6"]:
            if user.school_year in ["1", "2"]:
                return redirect(url_for('french_dictee'))
            if user.school_year in ["3", "4"]:
                return redirect(url_for('french_dictee_medium'))
    words_list = []
    with open("level-3-french.txt", "r", encoding="utf-8") as file:
        text = file.read()
        words_list = text.split('\n')
    exercise_list = random.sample(words_list, 3)
    if request.method == "POST":
        answer1 = request.form.get('q1')
        answer2 = request.form.get('q2')
        answer3 = request.form.get('q3')
        correct_answer1 = request.form.get('a1')
        correct_answer2 = request.form.get('a2')
        correct_answer3 = request.form.get('a2')
        score1 = 100 * (1 - levenshtein(correct_answer1, answer1) / len(correct_answer1))
        score2 = 100 * (1 - levenshtein(correct_answer2, answer2) / len(correct_answer2))
        score3 = 100 * (1 - levenshtein(correct_answer3, answer3) / len(correct_answer3))
        final_score = int((score1 + score2 + score3) / 15)
        exercise_attempt = ExerciseAttempt(user_id=session.get("user"), score=final_score, exercise_type="ENGLISH_DICTEE",
                                           exercise_level="HARD")
        db.session.add(exercise_attempt)
        db.session.commit()
        return render_template('application/result_exercise.html', score=final_score, user=user)
    return render_template('application/french_dictee.html', user=user, words=exercise_list)


@app.route('/arabic/reading/easy', methods=['GET', 'POST'])
def arabic_reading():
    if not session:
        return redirect(url_for('hello_world'))
    else:
        user = User.query.filter_by(id=session['user']).first()
        if user.role != "STUDENT":
            return redirect(url_for('hello_world'))
        if user.role == "STUDENT" and not user.test_taken:
            return redirect(url_for('test'))
        if user.role == "STUDENT" and user.school_year not in ["1", "2"]:
            if user.school_year in ["3", "4"]:
                return redirect(url_for('arabic_reading_medium'))
            if user.school_year in ["5", "6"]:
                return redirect(url_for('arabic_reading_hard'))
    word = ""
    with open("level-1-arabic.txt", "r", encoding="utf-8") as file:
        text = file.read()
        words_list = text.split('\n')
    word = random.choice(words_list)
    if request.method == "POST":
        answer = araby.strip_diacritics(request.form.get('a'))
        correct_answer = araby.strip_diacritics(request.form.get('ca'))
        score = int(20 * (1 - levenshtein(correct_answer, answer) / len(correct_answer)))
        exercise_attempt = ExerciseAttempt(user_id=session.get("user"), score=score, exercise_type="ARAB_READING",
                                           exercise_level="EASY")
        db.session.add(exercise_attempt)
        db.session.commit()
        return render_template('application/result_exercise.html', score=score, user=user)
    return render_template('application/pronunciation_arabic.html', user=user, word=word)


@app.route('/arabic/reading/medium', methods=['GET', 'POST'])
def arabic_reading_medium():
    if not session:
        return redirect(url_for('hello_world'))
    else:
        user = User.query.filter_by(id=session['user']).first()
        if user.role != "STUDENT":
            return redirect(url_for('hello_world'))
        if user.role == "STUDENT" and not user.test_taken:
            return redirect(url_for('test'))
        if user.role == "STUDENT" and user.school_year not in ["3", "4"]:
            if user.school_year in ["1", "2"]:
                return redirect(url_for('arabic_reading'))
            if user.school_year in ["5", "6"]:
                return redirect(url_for('arabic_reading_hard'))
    word = ""
    with open("level-2-arabic.txt", "r", encoding="utf-8") as file:
        text = file.read()
        words_list = text.split('\n')
    word = random.choice(words_list)
    if request.method == "POST":
        answer = araby.strip_diacritics(request.form.get('a'))
        correct_answer = araby.strip_diacritics(request.form.get('ca'))
        score = int(20 * (1 - levenshtein(correct_answer, answer) / len(correct_answer)))
        exercise_attempt = ExerciseAttempt(user_id=session.get("user"), score=score, exercise_type="ARAB_READING",
                                           exercise_level="MEDIUM")
        db.session.add(exercise_attempt)
        db.session.commit()
        return render_template('application/result_exercise.html', score=score, user=user)
    return render_template('application/pronunciation_arabic.html', user=user, word=word)


@app.route('/arabic/reading/hard', methods=['GET', 'POST'])
def arabic_reading_hard():
    if not session:
        return redirect(url_for('hello_world'))
    else:
        user = User.query.filter_by(id=session['user']).first()
        if user.role != "STUDENT":
            return redirect(url_for('hello_world'))
        if user.role == "STUDENT" and not user.test_taken:
            return redirect(url_for('test'))
        if user.role == "STUDENT" and user.school_year not in ["5", "6"]:
            if user.school_year in ["1", "2"]:
                return redirect(url_for('arabic_reading'))
            if user.school_year in ["3", "4"]:
                return redirect(url_for('arabic_reading_medium'))
    word = ""
    with open("level-3-arabic.txt", "r", encoding="utf-8") as file:
        text = file.read()
        words_list = text.split('\n')
    word = random.choice(words_list)
    if request.method == "POST":
        answer1 = araby.strip_diacritics(request.form.get('a'))
        correct_answer1 = araby.strip_diacritics(request.form.get('ca'))
        score = int(20 * (1 - levenshtein(correct_answer1, answer1) / len(correct_answer1)))
        exercise_attempt = ExerciseAttempt(user_id=session.get("user"), score=score, exercise_type="ARAB_READING",
                                           exercise_level="HARD")
        db.session.add(exercise_attempt)
        db.session.commit()
        return render_template('application/result_exercise.html', score=score, user=user)
    return render_template('application/pronunciation_arabic.html', user=user, word=word)


@app.route('/english/reading/easy', methods=['GET', 'POST'])
def english_reading():
    if not session:
        return redirect(url_for('hello_world'))
    else:
        user = User.query.filter_by(id=session['user']).first()
        if user.role != "STUDENT":
            return redirect(url_for('hello_world'))
        if user.role == "STUDENT" and not user.test_taken:
            return redirect(url_for('test'))
        if user.role == "STUDENT" and user.school_year not in ["1", "2"]:
            if user.school_year in ["3", "4"]:
                return redirect(url_for('english_reading_medium'))
            if user.school_year in ["5", "6"]:
                return redirect(url_for('english_reading_hard'))
    word = ""
    with open("level-1-english.txt", "r", encoding="utf-8") as file:
        text = file.read()
        words_list = text.split('\n')
    word = random.choice(words_list)
    if request.method == "POST":
        answer = request.form.get('a')
        correct_answer = request.form.get('ca')
        score = int(20 * (1 - levenshtein(correct_answer, answer) / len(correct_answer)))
        exercise_attempt = ExerciseAttempt(user_id=session.get("user"), score=score, exercise_type="ENGLISH_READING",
                                           exercise_level="EASY")
        db.session.add(exercise_attempt)
        db.session.commit()
        return render_template('application/result_exercise.html', score=score, user=user)
    return render_template('application/pronunciation_english.html', user=user, word=word)


@app.route('/english/reading/medium', methods=['GET', 'POST'])
def english_reading_medium():
    if not session:
        return redirect(url_for('hello_world'))
    else:
        user = User.query.filter_by(id=session['user']).first()
        if user.role != "STUDENT":
            return redirect(url_for('hello_world'))
        if user.role == "STUDENT" and not user.test_taken:
            return redirect(url_for('test'))
        if user.role == "STUDENT" and user.school_year not in ["3", "4"]:
            if user.school_year in ["1", "2"]:
                return redirect(url_for('english_reading'))
            if user.school_year in ["5", "6"]:
                return redirect(url_for('english_reading_hard'))
    word = ""
    with open("level-2-english.txt", "r", encoding="utf-8") as file:
        text = file.read()
        words_list = text.split('\n')
    word = random.choice(words_list)
    if request.method == "POST":
        answer = request.form.get('a')
        correct_answer = request.form.get('ca')
        score = int(20 * (1 - levenshtein(correct_answer, answer) / len(correct_answer)))
        exercise_attempt = ExerciseAttempt(user_id=session.get("user"), score=score, exercise_type="ENGLISH_READING",
                                           exercise_level="MEDIUM")
        db.session.add(exercise_attempt)
        db.session.commit()
        return render_template('application/result_exercise.html', score=score, user=user)
    return render_template('application/pronunciation_english.html', user=user, word=word)


@app.route('/english/reading/hard', methods=['GET', 'POST'])
def english_reading_hard():
    if not session:
        return redirect(url_for('hello_world'))
    else:
        user = User.query.filter_by(id=session['user']).first()
        if user.role != "STUDENT":
            return redirect(url_for('hello_world'))
        if user.role == "STUDENT" and not user.test_taken:
            return redirect(url_for('test'))
        if user.role == "STUDENT" and user.school_year not in ["5", "6"]:
            if user.school_year in ["1", "2"]:
                return redirect(url_for('english_reading'))
            if user.school_year in ["3", "4"]:
                return redirect(url_for('english_reading_medium'))
    word = ""
    with open("level-3-english.txt", "r", encoding="utf-8") as file:
        text = file.read()
        words_list = text.split('\n')
    word = random.choice(words_list)
    if request.method == "POST":
        answer = request.form.get('a')
        correct_answer = request.form.get('ca')
        score = int(20 * (1 - levenshtein(correct_answer, answer) / len(correct_answer)))
        exercise_attempt = ExerciseAttempt(user_id=session.get("user"), score=score, exercise_type="ENGLISH_READING",
                                           exercise_level="HARD")
        db.session.add(exercise_attempt)
        db.session.commit()
        return render_template('application/result_exercise.html', score=score, user=user)
    return render_template('application/pronunciation_english.html', user=user, word=word)


@app.route('/french/reading/easy', methods=['GET', 'POST'])
def french_reading():
    if not session:
        return redirect(url_for('hello_world'))
    else:
        user = User.query.filter_by(id=session['user']).first()
        if user.role != "STUDENT":
            return redirect(url_for('hello_world'))
        if user.role == "STUDENT" and not user.test_taken:
            return redirect(url_for('test'))
        if user.role == "STUDENT" and user.school_year not in ["1", "2"]:
            if user.school_year in ["3", "4"]:
                return redirect(url_for('french_reading_medium'))
            if user.school_year in ["5", "6"]:
                return redirect(url_for('french_reading_hard'))
    word = ""
    with open("level-1-french.txt", "r", encoding="utf-8") as file:
        text = file.read()
        words_list = text.split('\n')
    word = random.choice(words_list)
    if request.method == "POST":
        answer = request.form.get('a')
        correct_answer = request.form.get('ca')
        score = int(20 * (1 - levenshtein(correct_answer, answer) / len(correct_answer)))
        exercise_attempt = ExerciseAttempt(user_id=session.get("user"), score=score, exercise_type="FRENCH_READING",
                                           exercise_level="EASY")
        db.session.add(exercise_attempt)
        db.session.commit()
        return render_template('application/result_exercise.html', score=score, user=user)
    return render_template('application/pronunciation_french.html', user=user, word=word)


@app.route('/french/reading/medium', methods=['GET', 'POST'])
def french_reading_medium():
    if not session:
        return redirect(url_for('hello_world'))
    else:
        user = User.query.filter_by(id=session['user']).first()
        if user.role != "STUDENT":
            return redirect(url_for('hello_world'))
        if user.role == "STUDENT" and not user.test_taken:
            return redirect(url_for('test'))
        if user.role == "STUDENT" and user.school_year not in ["3", "4"]:
            if user.school_year in ["1", "2"]:
                return redirect(url_for('french_reading'))
            if user.school_year in ["5", "6"]:
                return redirect(url_for('french_reading_hard'))
    word = ""
    with open("level-2-french.txt", "r", encoding="utf-8") as file:
        text = file.read()
        words_list = text.split('\n')
    word = random.choice(words_list)
    if request.method == "POST":
        answer = request.form.get('a')
        correct_answer = request.form.get('ca')
        score = int(20 * (1 - levenshtein(correct_answer, answer) / len(correct_answer)))
        exercise_attempt = ExerciseAttempt(user_id=session.get("user"), score=score, exercise_type="FRENCH_READING",
                                           exercise_level="MEDIUM")
        db.session.add(exercise_attempt)
        db.session.commit()
        return render_template('application/result_exercise.html', score=score, user=user)
    return render_template('application/pronunciation_french.html', user=user, word=word)


@app.route('/french/reading/hard', methods=['GET', 'POST'])
def french_reading_hard():
    if not session:
        return redirect(url_for('hello_world'))
    else:
        user = User.query.filter_by(id=session['user']).first()
        if user.role != "STUDENT":
            return redirect(url_for('hello_world'))
        if user.role == "STUDENT" and not user.test_taken:
            return redirect(url_for('test'))
        if user.role == "STUDENT" and user.school_year not in ["5", "6"]:
            if user.school_year in ["1", "2"]:
                return redirect(url_for('french_reading'))
            if user.school_year in ["3", "4"]:
                return redirect(url_for('french_reading_medium'))
    word = ""
    with open("level-3-french.txt", "r", encoding="utf-8") as file:
        text = file.read()
        words_list = text.split('\n')
    word = random.choice(words_list)
    if request.method == "POST":
        answer = request.form.get('a')
        correct_answer = request.form.get('ca')
        score = int(20 * (1 - levenshtein(correct_answer, answer) / len(correct_answer)))
        exercise_attempt = ExerciseAttempt(user_id=session.get("user"), score=score, exercise_type="FRENCH_READING",
                                           exercise_level="HARD")
        db.session.add(exercise_attempt)
        db.session.commit()
        return render_template('application/result_exercise.html', score=score, user=user)
    return render_template('application/pronunciation_french.html', user=user, word=word)


@app.route('/arabic/letters', methods=['GET', 'POST'])
def arabic_letters():
    if not session:
        return redirect(url_for('hello_world'))
    user = User.query.filter_by(id=session['user']).first()
    if user.role != "STUDENT":
        return redirect(url_for('hello_world'))
    if not user.test_taken:
        return redirect(url_for('hello_world'))
    return render_template('application/arab_letters_exercise.html', user=user)


@app.route('/digits', methods=['GET', 'POST'])
def digits():
    if not session:
        return redirect(url_for('hello_world'))
    user = User.query.filter_by(id=session['user']).first()
    if user.role != "STUDENT":
        return redirect(url_for('hello_world'))
    if not user.test_taken:
        return redirect(url_for('hello_world'))
    return render_template('application/digits_exercise.html', user=user)


if __name__ == '__main__':
    app.run(debug=True)
