from django.urls import path
from . import views

urlpatterns = [
    # pages
    path('',              views.index,          name='home'),
    path('about/',        views.about,          name='about'),
    path('detect/',       views.detect,         name='detect'),
    path('contact/',      views.contact,        name='contact'),
    path('user_home/',    views.userhome,       name='user_home'),
    path('video-call/', views.video_call, name='video_call'),

    # webcam stream
    path('video_feed/',   views.video_feed,     name='video_feed'),

    # ── caption API endpoints ──────────────────────────────
    path('get_caption/',    views.get_caption,    name='get_caption'),
    path('clear_caption/',  views.clear_caption,  name='clear_caption'),
    path('get_suggestions/', views.get_suggestions, name='get_suggestions'),
    path('add_suggestion/', views.add_suggestion, name='add_suggestion'),
    path('get_voices/',    views.get_voices,    name='get_voices'),
    path('speak_caption/', views.speak_caption, name='speak_caption'),  # already exists, no change
    path('backspace_letter/', views.backspace_letter, name='backspace_letter'),

    # auth
    path('user_login/',        views.user_login,        name='user_login'),
    path('user_registration/', views.user_registration, name='user_registration'),
    path('user_logout/',       views.user_logout,       name='logout'),

    # otp
    path('otp_login_home/',                    views.otp_login_home, name='otp_login_home'),
    path('otp_login_home/verify_otp/',         views.verify_otp,     name='verify_otp'),
    path('otp_login_home/verify_otp/resend_OTP/', views.resend_otp,  name='resend_OTP'),

    path('delete_last_word/', views.delete_last_word, name='delete_last_word'),
    path('clear_buffer/',     views.clear_buffer,     name='clear_buffer'),
]
