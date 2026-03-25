from .models import *
from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
import cv2
import numpy as np
import time
import os
from gtts import gTTS
import io
from app.sign_detector import SignDetector
# ── GLOBAL STATE ──────────────────────────────────────────────
caption_text   = ""
letter_buffer  = ""
last_label     = ""
hold_start     = None
HOLD_SECONDS   = 2.0
last_confirmed = ""

# ── WORDLIST (NLTK frequency-ranked) ─────────────────────────
# import nltk
# from nltk.probability import FreqDist

# for _c in ['brown', 'words']:
#     try:
#         nltk.data.find(f'corpora/{_c}')
#     except LookupError:
#         nltk.download(_c, quiet=True)

# from nltk.corpus import brown as _brown, words as _nltk_words

# def _build_wordlist():
#     all_words   = [w.upper() for w in _brown.words() if w.isalpha() and 2 < len(w) < 12]
#     freq        = FreqDist(all_words)
#     freq_ranked = [w for w, _ in freq.most_common()]
#     dict_words  = set(w.upper() for w in _nltk_words.words() if w.isalpha() and 2 < len(w) < 12)
#     dict_only   = [w for w in dict_words if w not in set(freq_ranked)]
#     return freq_ranked + dict_only

# WORDLIST = _build_wordlist()


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

    cap      = cv2.VideoCapture(0)
    detector = SignDetector()

    # Match resolution used in 1_collect_data.py exactly
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    try:
        while True:
            success, img = cap.read()
            if not success:
                break

            # predict_frame handles: flip, mediapipe, landmarks, status bar, label overlay
            detected_label, confidence, img_output = detector.predict_frame(img)

            now = time.time()

            if detected_label:
                if detected_label == last_label:
                    if hold_start is None:
                        hold_start = now
                    held_for = now - hold_start
                    progress  = min(held_for / HOLD_SECONDS, 1.0)

                    # Timer ring top-right
                    from app.sign_detector import _draw_timer_ring
                    _draw_timer_ring(img_output, progress)

                    # Confirm after 2 seconds
                    if held_for >= HOLD_SECONDS and detected_label != last_confirmed:
                        last_confirmed = detected_label
                        letter_buffer += detected_label + " "
                        hold_start     = None
                        detector.reset_votes()
                else:
                    last_label     = detected_label
                    hold_start     = now
                    last_confirmed = ""
            else:
                last_label = ""
                hold_start = None

            # Caption bar at bottom
            h_f, w_f = img_output.shape[:2]
            cv2.rectangle(img_output, (0, h_f - 50), (w_f, h_f), (0, 0, 0), -1)
            display = (caption_text + (f"[{letter_buffer}]" if letter_buffer else ""))[-55:]
            cv2.putText(img_output, display, (10, h_f - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

            ret, buffer = cv2.imencode('.jpg', img_output)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        cap.release()
        detector.close()


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


# ── SMART WORD SUGGESTIONS ────────────────────────────────────
# Domain-specific sign language vocabulary — common everyday signs
# ── SMART WORD SUGGESTIONS ────────────────────────────────────
SIGN_VOCAB = [
    # Your trained signs
    "Hello", "Thanks", "Teacher", "Indian", "I am", "You are",
    "Beautiful", "Good", "Practice", "Man", "Woman", "Place",
    "Time", "Marry", "House", "Food",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",

    # Common day-to-day words not in your trained set
    "Yes", "No", "Please", "Sorry", "Welcome", "Goodbye", "Help",
    "Water", "Work", "School", "Hospital", "Name", "Family",
    "Mother", "Father", "Brother", "Sister", "Friend", "Baby",
    "Doctor", "Boy", "Girl", "Happy", "Sad", "Hungry", "Tired",
    "Morning", "Night", "Today", "Tomorrow", "Now", "Later",
    "Here", "There", "Come", "Go", "Wait", "Stop", "Start",
    "Eat", "Sleep", "Study", "Love", "Like", "Want", "Need",
    "Big", "Small", "Hot", "Cold", "New", "Old", "Fast", "Slow",
    "Money", "Bus", "Car", "Market", "City", "Village", "Country",
    "Phone", "Book", "Clothes", "Tea", "Rice", "Milk",
]

NEXT_WORD_MAP = {
    # After your trained signs
    "I AM":       ["Good", "Fine", "Happy", "Tired", "Hungry", "Here",
                   "Sorry", "Ready", "Indian", "Your Teacher", "A Man", "A Woman"],
    "YOU ARE":    ["Good", "Beautiful", "Welcome", "My Friend", "A Teacher",
                   "Indian", "Happy", "The Best"],
    "HELLO":      ["Friend", "Teacher", "Brother", "Sister", "Everyone",
                   "Good Morning", "How Are You"],
    "THANKS":     ["Brother", "Sister", "Friend", "Teacher", "So Much"],
    "GOOD":       ["Morning", "Night", "Food", "Work", "Practice",
                   "Time", "Place", "Man", "Woman", "Teacher"],
    "BEAUTIFUL":  ["Place", "Woman", "House", "Day", "Time", "Food"],
    "PRACTICE":   ["Now", "Every Day", "Sign", "More", "Together"],
    "MARRY":      ["Me", "Now", "Soon", "Him", "Her"],
    "INDIAN":     ["Man", "Woman", "Food", "Teacher", "House", "Place"],
    "TEACHER":    ["Is Good", "Is Here", "Teaches", "Helps"],
    "FOOD":       ["Is Good", "Is Ready", "Is Hot", "Is Here"],
    "HOUSE":      ["Is Big", "Is Old", "Is New", "Is Here", "Is Beautiful"],
    "PLACE":      ["Is Good", "Is Beautiful", "Is Far", "Is Near"],
    "TIME":       ["Is Now", "Is Good", "For Food", "For Work", "For Practice"],
    "MAN":        ["Is Good", "Is Indian", "Is Teacher", "Is Here"],
    "WOMAN":      ["Is Good", "Is Beautiful", "Is Indian", "Is Teacher"],

    # Common follow-ups for extra vocab
    "YES":        ["Please", "I Want", "Good", "Now", "Come"],
    "NO":         ["Thanks", "Please", "Not Now", "Sorry"],
    "PLEASE":     ["Help", "Come", "Wait", "Give", "Stop"],
    "SORRY":      ["Friend", "Teacher", "Please", "I Am Late"],
    "HELP":       ["Me", "Please", "Now", "Here"],
    "COME":       ["Here", "Home", "School", "Hospital", "Now"],
    "GO":         ["Home", "School", "Hospital", "Now", "Together"],
    "WANT":       ["Food", "Water", "Help", "To Go", "To Come", "To Marry"],
    "NEED":       ["Help", "Food", "Water", "Time", "Practice"],
    "EAT":        ["Food", "Now", "Together", "Rice", "Here"],
    "MOTHER":     ["Is Good", "Is Beautiful", "Is Here", "Loves Me"],
    "FATHER":     ["Is Good", "Is Here", "Works Hard"],
    "FRIEND":     ["Is Good", "Is Here", "Helps Me", "Is Indian"],
    "HAPPY":      ["Now", "Today", "With You", "To Be Here"],
    "HUNGRY":     ["Now", "Please", "Help", "Feed Me"],
    "MORNING":    ["Is Good", "Everyone", "Today"],
    "TODAY":      ["Is Good", "Practice", "Time", "Work", "Food"],
    "TOMORROW":   ["Practice", "Work", "Come", "Go", "Time"],
    "WATER":      ["Please", "Is Cold", "Is Hot", "Now"],
    "SCHOOL":     ["Is Good", "Is Here", "Has Teacher", "Practice"],
    "FAMILY":     ["Is Good", "Is Here", "Is Indian", "Loves Me"],
    "DOCTOR":     ["Is Good", "Is Here", "Helps Me"],
    "LOVE":       ["You", "Food", "Practice", "My Family", "Indian"],
}
def get_suggestions(request):
    prefix = letter_buffer.strip()
    if not prefix:
        return JsonResponse({"suggestions": []})

    prefix_upper = prefix.upper().strip()

    # Priority 1 — next-word context suggestions
    # Look at the last confirmed word in caption_text
    last_word = caption_text.strip().split()[-1].upper() if caption_text.strip() else ""
    context_suggestions = []
    if last_word and last_word in NEXT_WORD_MAP:
        context_suggestions = [
            w for w in NEXT_WORD_MAP[last_word]
            if w.upper().startswith(prefix_upper)
        ]

    # Also check last two words (e.g. "I AM" as key)
    words = caption_text.strip().split()
    if len(words) >= 2:
        last_two = (words[-2] + " " + words[-1]).upper()
        if last_two in NEXT_WORD_MAP:
            context_suggestions += [
                w for w in NEXT_WORD_MAP[last_two]
                if w.upper().startswith(prefix_upper)
                and w not in context_suggestions
            ]

    # Priority 2 — domain vocab prefix match
    vocab_suggestions = [
        w for w in SIGN_VOCAB
        if w.upper().startswith(prefix_upper)
        and w not in context_suggestions
    ]

    # Combine: context first, then vocab, max 6
    suggestions = (context_suggestions + vocab_suggestions)[:6]

    return JsonResponse({"suggestions": suggestions, "prefix": prefix})


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
    # Speak whatever is visible — confirmed caption + current buffer
    text = (caption_text + " " + letter_buffer).strip()
    voice_id = request.GET.get("voice", "en-us-f")

    if not text:
        return JsonResponse({"status": "empty"})

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
    except Exception:
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


@csrf_exempt
def delete_last_word(request):
    """Z key — removes the last word from caption_text."""
    global caption_text
    words = caption_text.strip().split()
    if words:
        words.pop()
        caption_text = " ".join(words) + (" " if words else "")
    return JsonResponse({"status": "ok", "caption": caption_text})


@csrf_exempt
def clear_buffer(request):
    """Clears only letter_buffer without touching caption_text."""
    global letter_buffer
    letter_buffer = ""
    return JsonResponse({"status": "ok"})