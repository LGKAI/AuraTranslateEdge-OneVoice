# -*- coding: utf-8 -*-
"""fix_itn (English clock times that SenseVoice + greedy CTC garble). CHANGE = the two real BTC cases and their direct variants;
KEEP = numbers that are not times, or times that cannot be told apart from numbers. Every case here is a pattern test, not a sentence lookup."""
import os, random, string, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s2s_edge.asr import fix_itn

CHANGE = {
    "Get me a taxi to take me to the airport on Friday, at530 AM.": "Get me a taxi to take me to the airport on Friday, at 5:30 AM.",   # BTC EV-3775
    "Your appointment has been scheduled for Wednesday at 8 30 am.": "Your appointment has been scheduled for Wednesday at 8:30 am.",           # BTC EV-6709
    "Wake me at 530 AM": "Wake me at 5:30 AM", "The meeting is at 1230 PM": "The meeting is at 12:30 PM", "at 700 AM": "at 7:00 AM",
    "from 530 AM to 630 PM": "from 5:30 AM to 6:30 PM", "after 830 a.m.": "after 8:30 a.m.", "around 10 45 pm": "around 10:45 pm", "by 1000 AM": "by 10:00 AM",
    "dividing by2 to obtain": "dividing by 2 to obtain", "It was2024": "It was 2024",                                                          # glued digits only get their space
}
KEEP = [
    "at 5:30 AM", "at 8:30 am", "Tell me, what was the temperature in our city at 12 PM.", "scheduled for Thursday at 8 a.m, is there anything else",
    "I have 1000 AM radios", "Room 530 AM building", "Is 530 PM ok", "the 745 PM show", "We had 2530 AM listeners",
    "Call me at 1330 PM", "at 2330 pm", "at 0 30 am", "at 8 75 am", "at 800.", "at 1530", "Conver 2330 from GMMT plus 430 to GMT 200.",
    "Play only all music released between the year 1990 and 2000.", "734 Main Street, Harper Valley Oregon 67297.", "pay 900 AM", "at 5 30 AMD",
    "I like it", "", "no digits here AM PM",
]

def test_change():
    for a, b in CHANGE.items(): assert fix_itn(a, "en") == b, (a, fix_itn(a, "en"), b)

def test_keep():
    for a in KEEP: assert fix_itn(a, "en") == a or a.replace("at530", "at 530") == fix_itn(a, "en"), (a, fix_itn(a, "en"))

def test_other_languages_untouched():
    for lang in ("vi", "zh", "ko"): assert fix_itn("at530 AM", lang) == "at530 AM"

def test_idempotent_and_insert_only():
    rnd = random.Random(1); alpha = string.ascii_letters + "   0123456789:.,-"; words = ["at", "by", "to", "from", "AM", "pm", "a.m.", "530", "8", "30", "1230", "x", ":"]
    for _ in range(20000):
        t = " ".join(rnd.choice(words) if rnd.random() < .7 else "".join(rnd.choice(alpha) for _ in range(rnd.randint(1, 5))) for _ in range(rnd.randint(1, 8)))
        o = fix_itn(t, "en"); assert fix_itn(o, "en") == o, (t, o)
        assert o.replace(":", "").replace(" ", "") == t.replace(":", "").replace(" ", ""), (t, o)       # only ":" and " " are inserted, nothing is removed or reordered
        if not any(c.isdigit() for c in t): assert o == t

if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("ok", k)
