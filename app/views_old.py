from .models import *
from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
import cv2
import numpy as np
import math
import time
import os
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from gtts import gTTS
import io
# ── GLOBAL STATE ──────────────────────────────────────────────
caption_text   = ""
letter_buffer  = ""
last_label     = ""
hold_start     = None
HOLD_SECONDS   = 2.0
last_confirmed = ""

# ── WORDLIST (NLTK frequency-ranked) ─────────────────────────
import nltk
from nltk.probability import FreqDist

for _c in ['brown', 'words']:
    try:
        nltk.data.find(f'corpora/{_c}')
    except LookupError:
        nltk.download(_c, quiet=True)

from nltk.corpus import brown as _brown, words as _nltk_words

def _build_wordlist():
    all_words   = [w.upper() for w in _brown.words() if w.isalpha() and 2 < len(w) < 12]
    freq        = FreqDist(all_words)
    freq_ranked = [w for w, _ in freq.most_common()]
    dict_words  = set(w.upper() for w in _nltk_words.words() if w.isalpha() and 2 < len(w) < 12)
    dict_only   = [w for w in dict_words if w not in set(freq_ranked)]
    return freq_ranked + dict_only

WORDLIST = _build_wordlist()


# ── PAGE VIEWS ────────────────────────────────────────────────
def index(request):    return render(request, 'index.html')
def about(request):    return render(request, 'about.html')
def detect(request):
    if not request.session.get('user_id'):
        messages.error(request, "Please log in to use the detection feature.")
        return redirect('user_login')
    return render(request, 'detect.html')
def contact(request):  return render(request, 'contact.html')
def userhome(request): return render(request, 'user_home.html')
def predict(request):  return render(request, 'predict.html')


# ── FRAME GENERATOR ───────────────────────────────────────────
def generate_frames():
    global caption_text, letter_buffer, last_label, hold_start, last_confirmed

    cap        = cv2.VideoCapture(0)
    detector   = HandDetector(maxHands=1)
    classifier = Classifier("model/keras_model.h5", "model/labels.txt")
    offset     = 20
    img_size   = 300

    # A–Z only — matches your trained model exactly
    labels = [
    "I am",
    "You are",
    "Teacher",
    "Thanks",
    "Clever",
    "L",
    "O",
    "M",
    "V",
    "Indian",
    "Hello",
    "Beautiful",
    "Yes",
    "Women",
    "E",
    "Day",
]

    while True:
        success, img = cap.read()
        if not success:
            break

        img        = cv2.resize(img, (640, 480))
        img_output = img.copy()
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            img_white = np.ones((img_size, img_size, 3), np.uint8) * 255

            y1 = max(0, y - offset);  y2 = min(img.shape[0], y + h + offset)
            x1 = max(0, x - offset);  x2 = min(img.shape[1], x + w + offset)
            img_crop = img[y1:y2, x1:x2]

            if img_crop.size != 0:
                ar = h / w if w != 0 else 1
                if ar > 1:
                    k     = img_size / h
                    w_cal = math.ceil(k * w)
                    w_gap = math.ceil((img_size - w_cal) / 2)
                    img_white[:, w_gap:w_cal + w_gap] = cv2.resize(img_crop, (w_cal, img_size))
                else:
                    k     = img_size / w
                    h_cal = math.ceil(k * h)
                    h_gap = math.ceil((img_size - h_cal) / 2)
                    img_white[h_gap:h_cal + h_gap, :] = cv2.resize(img_crop, (img_size, h_cal))

                prediction, index = classifier.getPrediction(img_white)
                if index < len(labels):
                    detected_label = labels[index]
                    confidence     = prediction[index]
                    now            = time.time()

                    if detected_label == last_label:
                        if hold_start is None:
                            hold_start = now
                        held_for = now - hold_start

                        # Progress arc
                        progress  = min(held_for / HOLD_SECONDS, 1.0)
                        cx, cy    = x + w // 2, y - 40
                        angle_end = int(-90 + 360 * progress)
                        cv2.ellipse(img_output, (cx, cy), (30, 30), 0, -90, angle_end, (0, 255, 150), 3)
                        cv2.putText(img_output, f"{round(held_for, 1)}s", (cx - 18, cy + 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 150), 1)

                        # Confirm on 2s hold
                        if held_for >= HOLD_SECONDS and detected_label != last_confirmed:
                            last_confirmed  = detected_label
                            letter_buffer  += detected_label   # always add letter to buffer
                            hold_start      = None
                    else:
                        last_label     = detected_label
                        hold_start     = now
                        last_confirmed = ""

                    conf_pct = int(confidence * 100)
                    cv2.putText(img_output, f"{detected_label}  {conf_pct}%",
                                (x1, max(y1 - 15, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
                    cv2.rectangle(img_output, (x1, y1), (x2, y2), (255, 0, 255), 2)

        # Caption bar on frame
        h_f, w_f = img_output.shape[:2]
        cv2.rectangle(img_output, (0, h_f - 50), (w_f, h_f), (0, 0, 0), -1)
        display = (caption_text + (f"[{letter_buffer}]" if letter_buffer else ""))[-55:]
        cv2.putText(img_output, display, (10, h_f - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', img_output)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()


def video_feed(request):
    return StreamingHttpResponse(generate_frames(),
                                 content_type='multipart/x-mixed-replace; boundary=frame')


# ── CAPTION API ───────────────────────────────────────────────
def get_caption(request):
    return JsonResponse({
        "caption": caption_text,
        "buffer":  letter_buffer,
        "display": caption_text + (f"[{letter_buffer}]" if letter_buffer else ""),
    })


@csrf_exempt
def clear_caption(request):
    global caption_text, letter_buffer, last_confirmed, hold_start
    caption_text = letter_buffer = last_confirmed = ""
    hold_start   = None
    return JsonResponse({"status": "ok"})


@csrf_exempt
def add_suggestion(request):
    global caption_text, letter_buffer
    word = request.POST.get("word", "").strip().upper()
    if word:
        caption_text += word + " "
        letter_buffer = ""
    return JsonResponse({"status": "ok", "caption": caption_text})


@csrf_exempt
def commit_word(request):
    """SPACE gesture — commits letter_buffer as a full word."""
    global caption_text, letter_buffer
    if letter_buffer.strip():
        caption_text += letter_buffer.strip() + " "
        letter_buffer = ""
    return JsonResponse({"status": "ok", "caption": caption_text})


def get_suggestions(request):
    prefix = letter_buffer.upper().strip()
    if len(prefix) < 1:
        return JsonResponse({"suggestions": []})
    word_matches = [w for w in WORDLIST if w.startswith(prefix)][:6]
    return JsonResponse({"suggestions": word_matches, "prefix": prefix})




def get_voices(request):
    """Returns available voice options."""
    voices = [
        {"id": "en-us-f", "label": "English – US Female",     "lang": "en", "tld": "com"},
        {"id": "en-us-m", "label": "English – US Male",        "lang": "en", "tld": "com.au"},
        {"id": "en-uk",   "label": "English – UK",             "lang": "en", "tld": "co.uk"},
        {"id": "en-in",   "label": "English – Indian",         "lang": "en", "tld": "co.in"},
        {"id": "en-au",   "label": "English – Australian",     "lang": "en", "tld": "com.au"},
        {"id": "en-ca",   "label": "English – Canadian",       "lang": "en", "tld": "ca"},
    ]
    return JsonResponse({"voices": voices})


def speak_caption(request):
    text     = caption_text.strip()
    voice_id = request.GET.get("voice", "en-us-f")

    if not text:
        return JsonResponse({"status": "empty"})

    # Map voice_id to gTTS params
    voice_map = {
        "en-us-f": {"lang": "en", "tld": "com"},
        "en-us-m": {"lang": "en", "tld": "com.au"},
        "en-uk":   {"lang": "en", "tld": "co.uk"},
        "en-in":   {"lang": "en", "tld": "co.in"},
        "en-au":   {"lang": "en", "tld": "com.au"},
        "en-ca":   {"lang": "en", "tld": "ca"},
    }

    params = voice_map.get(voice_id, {"lang": "en", "tld": "com"})

    try:
        from django.http import HttpResponse
        tts    = gTTS(text=text, lang=params["lang"], tld=params["tld"], slow=False)
        buffer = io.BytesIO()
        tts.write_to_fp(buffer)
        buffer.seek(0)
        return HttpResponse(buffer.read(), content_type="audio/mpeg")
    except Exception as e:
        # Fallback to browser TTS if gTTS fails (no internet etc.)
        return JsonResponse({"status": "fallback", "text": text})

@csrf_exempt
def backspace_letter(request):
    global letter_buffer
    if letter_buffer:
        letter_buffer = letter_buffer[:-1]
    return JsonResponse({"status": "ok", "buffer": letter_buffer})


# ── AUTH (unchanged) ──────────────────────────────────────────
def user_login(request):
    if request.method == 'POST':
        user_email = request.POST.get('email')
        password   = request.POST.get('password')
        user = User.objects.filter(username=user_email, password=password).first()
        if not user:
            user = User.objects.filter(email=user_email, password=password).first()
        if user:
            request.session['user_id']  = user.id
            request.session['username'] = user.username
            messages.success(request, "Welcome back!")
            return redirect('user_home')
        messages.error(request, "Invalid credentials.")
        return redirect('user_login')
    return render(request, 'login.html')


def user_registration(request):
    if request.method == 'POST':
        username         = request.POST.get('username')
        name             = request.POST.get('name')
        email            = request.POST.get('email')
        password         = request.POST.get('password')
        confirm_password = request.POST.get('confirm_password')
        if password != confirm_password:
            messages.error(request, "Passwords do not match.")
            return redirect('user_registration')
        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already exists.")
            return redirect('user_registration')
        if User.objects.filter(email=email).exists():
            messages.error(request, "Email already registered.")
            return redirect('user_registration')
        User(username=username, name=name, email=email, password=password).save()
        messages.success(request, "Account created successfully!")
        return redirect('user_login')
    return render(request, 'register.html')


def user_logout(request):
    request.session.flush()
    messages.success(request, "Logged out successfully.")
    return redirect('home')


from django.core.mail import send_mail
from django.conf import settings
import random

otp_storage = {}

def send_otp(request, email):
    otp = random.randint(100000, 999999)
    otp_storage[email] = {"otp": otp, "time": time.time()}
    request.session['otp_email'] = email
    send_mail("Your OTP", f"OTP: {otp}. Valid 10 minutes.",
              settings.EMAIL_HOST_USER, [email])

def otp_login_home(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        if User.objects.filter(email=email).exists():
            send_otp(request, email)
            messages.success(request, f"OTP sent to {email}")
            return redirect('verify_otp')
        messages.error(request, "Email not registered.")
        return redirect('otp_login')
    return render(request, 'OTP/otp_login_home.html')

def verify_otp(request):
    if request.method == "POST":
        otp   = request.POST.get("otp")
        email = request.session.get("otp_email")
        now   = time.time()
        if not email or email not in otp_storage:
            messages.error(request, "No OTP. Request again.")
            return redirect('otp_login_home')
        stored = otp_storage[email]
        if now - stored["time"] > 600:
            del otp_storage[email]
            messages.error(request, "OTP expired.")
            return redirect('otp_login')
        if int(otp) == stored["otp"]:
            user = User.objects.filter(email=email).first()
            request.session['user_id'] = user.id
            del otp_storage[email]
            del request.session['otp_email']
            messages.success(request, "Logged in successfully")
            return redirect('user_home')
        messages.error(request, "Invalid OTP")
        return redirect('verify_otp')
    return render(request, 'OTP/verify_otp.html')

def resend_otp(request):
    email = request.session.get("otp_email")
    if not email:
        messages.error(request, "No email found.")
        return redirect('otp_login')
    otp = random.randint(100000, 999999)
    otp_storage[email] = {"otp": otp, "time": time.time()}
    send_mail("Your OTP (Resent)", f"New OTP: {otp}. Valid 10 minutes.",
              settings.EMAIL_HOST_USER, [email])
    messages.success(request, f"New OTP sent to {email}")
    return redirect('verify_otp')

def admin_login(request):
    if request.method == 'POST':
        u = request.POST.get('admin_username')
        p = request.POST.get('password')
        admin = AdminData.objects.filter(admin_username=u, password=p).first()
        if not admin:
            admin = AdminData.objects.filter(admin_email=u, password=p).first()
        if admin:
            request.session['admin_username'] = admin.admin_username
            messages.success(request, "Logged in")
            return redirect('admin_home')
        messages.error(request, "Invalid credentials.")
        return redirect('admin_login')
    return render(request, 'admin_login.html')

def admin_home(request):    return render(request, 'admin_dir/admin_home.html')
def admin_logout(request):  request.session.flush(); return redirect('home')

def user_contact(request):
    if request.method == 'POST':
        Contact(names=request.POST['names'], email=request.POST['email'],
                phone=request.POST['phone'], desc=request.POST['desc']).save()
        return redirect('contact')
    return render(request, 'contact.html')

def view_user(request):
    return render(request, 'view_user.html', {'forms': User.objects.all()})

def view_contact(request):
    return render(request, 'view_contact.html', {'forms': Contact.objects.all()})

def video_call(request):
    """Redirect logged-in users to MiroTalk C2C."""
    if not request.session.get('user_id'):
        messages.error(request, "Please log in to access video calling.")
        return redirect('user_login')
    # MiroTalk running on port 8080
    # Generate a room name from username so each user gets a unique room
    username = request.session.get('username', 'user')
    room = f"{username}-room"
    mirotalk_url = f"http://localhost:8080/?room={room}"
    return redirect(mirotalk_url)