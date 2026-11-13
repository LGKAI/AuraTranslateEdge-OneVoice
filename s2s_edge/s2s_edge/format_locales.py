"""Declarative spoken-form resources; numeric values are never stored here."""
DIGITS = {
    'vi': 'không một hai ba bốn năm sáu bảy tám chín'.split(),
    'en': 'zero one two three four five six seven eight nine'.split(),
    'ko': list('영일이삼사오육칠팔구'), 'zh': list('零一二三四五六七八九'),
}
LETTERS = {
    'vi': ['a','bê','xê','đê','e','ép','giê','hát','i','giây','ca','en lờ','em','en','ô','pê','quy','a rờ','ét','tê','u','vê','đắp liu','ích','i dài','dét'],
    'en': ['ay','bee','see','dee','ee','ef','gee','aitch','eye','jay','kay','el','em','en','oh','pee','cue','ar','ess','tee','you','vee','double you','ex','why','zee'],
    'ko': ['에이','비','씨','디','이','에프','지','에이치','아이','제이','케이','엘','엠','엔','오','피','큐','알','에스','티','유','브이','더블유','엑스','와이','지'],
    'zh': ['诶','比','西','迪','伊','艾弗','吉','艾尺','艾','杰','开','艾勒','艾姆','恩','欧','屁','丘','阿尔','艾丝','提','优','维','达不溜','艾克斯','歪','贼德'],
}
IDENTIFIER_CUES = (
    'phone', 'telephone', 'mobile', 'zip', 'postal', 'postcode', 'flight', 'call',
    'otp', 'pin', 'serial', 'tracking', 'booking', 'account number', 'order number',
    'điện thoại', 'di động', 'mã', 'bưu chính', 'chuyến bay', 'số tài khoản', 'gọi số',
    '전화번호', '우편번호', '항공편', '인증번호', '일련번호', '계좌번호',
    '电话号码', '手机号', '邮政编码', '邮编', '航班', '验证码', '订单号', '账号', '拨打', '致电',
)
CURRENCY = {
    'USD': {'vi': ('đô la', 'xen'), 'en': ('dollar', 'cent'), 'ko': ('달러', '센트'), 'zh': ('美元', '美分')},
    'EUR': {'vi': ('euro', 'xen'), 'en': ('euro', 'cent'), 'ko': ('유로', '센트'), 'zh': ('欧元', '欧分')},
    'GBP': {'vi': ('bảng Anh', 'pen ni'), 'en': ('pound', 'penny'), 'ko': ('파운드', '펜스'), 'zh': ('英镑', '便士')},
    'VND': {'vi': ('đồng', None), 'en': ('dong', None), 'ko': ('동', None), 'zh': ('越南盾', None)},
    'KRW': {'vi': ('won', None), 'en': ('won', None), 'ko': ('원', None), 'zh': ('韩元', None)},
    'CNY': {'vi': ('nhân dân tệ', 'phân'), 'en': ('yuan', 'fen'), 'ko': ('위안', '펀'), 'zh': ('元', '分')},
}
CURRENCY_ALIASES = {
    '$': 'USD', 'US$': 'USD', 'USD': 'USD', 'dollars': 'USD', 'dollar': 'USD', 'đô la': 'USD', '달러': 'USD', '美元': 'USD',
    '€': 'EUR', 'EUR': 'EUR', 'euros': 'EUR', 'euro': 'EUR', '유로': 'EUR', '欧元': 'EUR',
    '£': 'GBP', 'GBP': 'GBP', 'pounds': 'GBP', 'pound': 'GBP',
    '₫': 'VND', 'VND': 'VND', 'đồng': 'VND',
    '₩': 'KRW', 'KRW': 'KRW', '원': 'KRW',
    'CNY': 'CNY', '人民币': 'CNY', '元': 'CNY',
}
UNITS = {
    '°c': ('độ xê', 'degrees Celsius', '도', '摄氏度'),
    '℃': ('độ xê', 'degrees Celsius', '도', '摄氏度'),
    '°f': ('độ ép', 'degrees Fahrenheit', '화씨 도', '华氏度'),
    '°': ('độ', 'degrees', '도', '度'),
    'kg': ('ki lô gam', 'kilograms', '킬로그램', '千克'),
    'km': ('ki lô mét', 'kilometers', '킬로미터', '千米'),
    'km/h': ('ki lô mét trên giờ', 'kilometers per hour', '킬로미터 매 시', '千米每小时'),
    'm/s': ('mét trên giây', 'meters per second', '미터 매 초', '米每秒'),
    'cm': ('xen ti mét', 'centimeters', '센티미터', '厘米'),
    'mm': ('mi li mét', 'millimeters', '밀리미터', '毫米'),
    'm': ('mét', 'meters', '미터', '米'),
    'g': ('gam', 'grams', '그램', '克'),
    'ml': ('mi li lít', 'milliliters', '밀리리터', '毫升'),
    'l': ('lít', 'liters', '리터', '升'),
}
MONTHS_EN = 'January February March April May June July August September October November December'.split()
KO_HOURS = ['영', '한', '두', '세', '네', '다섯', '여섯', '일곱', '여덟', '아홉', '열', '열한', '열두']
