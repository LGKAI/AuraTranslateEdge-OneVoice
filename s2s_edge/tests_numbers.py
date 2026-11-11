# -*- coding: utf-8 -*-
import sys
from s2s_edge.textnorm import vi_num, zh_num, ko_num, en_num, normalize
sys.stdout.reconfigure(encoding="utf-8")
for name, f, xs in (("vi", vi_num, [0, 5, 10, 15, 21, 25, 105, 110, 305, 1000, 2024, 1001, 1234567, 2000000000]),
                    ("zh", zh_num, [0, 5, 10, 15, 21, 105, 110, 1000, 2024, 10086, 100000, 100005, 1234567]),
                    ("ko", ko_num, [0, 5, 10, 21, 105, 1000, 2024, 10086]), ("en", en_num, [0, 15, 21, 105, 1000, 2024, 1234567])):
    print(name, [(x, f(x)) for x in xs])
print(normalize("Có 25% người và 3,5 km, tổng 1.200 người", "vi")); print(normalize("温度是23.5度,增加了15%", "zh")); print(normalize("We have 1,200 workers and 7 shifts", "en")); print(normalize("총 1,200명 중 25%", "ko"))
