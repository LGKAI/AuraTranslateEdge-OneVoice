"""Three-page Vietnamese status/next-steps report from saved evaluation artifacts."""
from pathlib import Path
import json
from xml.sax.saxutils import escape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, KeepTogether
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'results/hybrid_format'
stats = json.loads((DATA/'evaluation.json').read_text())
assert stats['semantic_examples'] == {'passed':55, 'total':55}
assert stats['verified_audio_count'] == 59
assert stats['tts_replay_status'] == {'ok':49, 'skipped_empty_mt':1}
assert len(json.loads((DATA/'audio_crosscheck.json').read_text())['cases']) == 7
fonts = Path('/Users/khoa/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/libreoffice-headless/libreoffice/LibreOfficeDev.app/Contents/Resources/fonts/truetype')
for name, file in [('Body','DejaVuSans.ttf'), ('Bold','DejaVuSans-Bold.ttf')]:
    pdfmetrics.registerFont(TTFont(name, str(fonts/file)))
pdfmetrics.registerFontFamily('Body', normal='Body', bold='Bold', italic='Body', boldItalic='Bold')
NAVY = HexColor('#142F43'); TEAL = HexColor('#087F84'); GRAY = HexColor('#52616D')
styles = {
    'title':ParagraphStyle('title', fontName='Bold', fontSize=24, leading=30, textColor=NAVY, spaceAfter=10),
    'subtitle':ParagraphStyle('subtitle', fontName='Body', fontSize=9, leading=14, textColor=GRAY, spaceAfter=13),
    'section':ParagraphStyle('section', fontName='Bold', fontSize=12, leading=17, textColor=TEAL, spaceBefore=11, spaceAfter=6),
    'body':ParagraphStyle('body', fontName='Body', fontSize=9.3, leading=14.5, textColor=NAVY, spaceAfter=7),
    'small':ParagraphStyle('small', fontName='Body', fontSize=8, leading=12, textColor=GRAY, spaceAfter=5),
    'callout':ParagraphStyle('callout', fontName='Body', fontSize=10, leading=15, textColor=NAVY,
        backColor=HexColor('#EBF5F4'), borderPadding=10, spaceBefore=6, spaceAfter=14),
}
story = []
def p(text, style='body'): story.append(Paragraph(text, styles[style]))
def h(text): p(text, 'section')
def item(label, body): p('<b>'+label+'</b> '+body)
def new(): story.append(PageBreak())

p('Format TTS hybrid<br/>Đã làm được &amp; việc tiếp theo', 'title')
p('ONEVOICE | VI / EN / KO / ZH | Báo cáo trạng thái ngày 12/11/2026', 'subtitle')
p('<b>Kết luận:</b> Đã sửa cơ chế tạo văn bản đầu vào TTS và tích hợp vào pipeline. Các ca kiểm tra format đã qua; chất lượng phát âm vẫn cần người nghe xác nhận. Không coi test tạo WAV thành công là bằng chứng mọi nội dung được đọc đúng.', 'callout')
h('1. Cơ chế đã triển khai')
item('Luồng chung:', 'Bản dịch → nhận diện loại dữ liệu → giữ giá trị trong cấu trúc có kiểu → sinh lời đọc theo ngôn ngữ → chia đoạn → TTS.')
item('Không vá riêng chuỗi:', 'Cùng một grammar xử lý các giá trị giờ, ngày, tiền, phần trăm, số đo và số thập phân. Mã được đọc từng ký tự, giữ số 0 đầu. Không để model tự sinh lại số.')
item('Phần hybrid:', 'Grammar và ngữ cảnh nhận loại dữ liệu; CRF hỗ trợ thêm nhận diện mã ở Anh/Việt. Hàn/Trung hiện dùng grammar và từ vựng ngữ cảnh, chưa có CRF riêng. [N1, N2]')
h('2. Kết quả theo ngôn ngữ - tại đầu vào TTS')
item('Việt:', '5:30 → “năm giờ ba mươi phút”; 08:00 → “tám giờ”. ZIP 67297 đọc từng số; số nhà 734 giữ “bảy trăm ba mươi bốn”, khắc phục lỗi model đổi sai giá trị.')
item('Anh:', 'Giờ/ngày có cấu trúc lời đọc; $18.25 có dollars/cents; điện thoại, ZIP và mã giữ từng ký tự. Câu thử tổng hợp không còn phụ thuộc LLM sinh span.')
item('Hàn:', 'Giờ dùng cách đếm phù hợp và đơn vị giờ/phút; mã đọc từng ký tự. Ngân sách graph được tính sau khi tách Hangul thành Jamo (NFKD).')
item('Trung:', 'Giờ có đơn vị giờ/phút; năm đọc từng chữ số; mã đọc từng ký tự. Đã sửa cả thang số lớn 10^12 trong codec. [N2, N3]')
h('3. Tích hợp và hiệu năng')
p('Đặt <b>S2S_TTS_FORMAT=hybrid</b> làm mặc định; giữ legacy/neural để đối chiếu. Đã cập nhật web localhost và test upload audio thật, không mở Chrome, chưa test mic vật lý.')
p('Format CPU sau warm-up: <b>trung bình 0,23 ms; median 0,17 ms; p95 0,47 ms</b> trên 196 lượt. Chỉ tính chuẩn hóa, không gồm ASR/MT/G2P/TTS. Code TN mới khoảng 27,5 KB; CRF có sẵn 62 KB. Đây không phải RAM hoặc toàn bộ dung lượng model. [N1]','small')

new()
p('Kiểm thử &amp; đánh giá thực tế', 'title')
p('Phân biệt: đúng format văn bản / chạy hết pipeline / nghe đúng nội dung.', 'subtitle')
h('4. Đã kiểm tra những gì?')
item('55/55 ví dụ kỳ vọng:', 'Giờ, trưa/đêm, ngày, tiền, số âm/thập phân, %, đơn vị, mã, số 0 đầu, Unicode fullwidth và số lớn Trung.')
item('1.600 ca sinh tự động:', 'Kiểm tra cấu trúc giá trị của giờ/ngày/thập phân/mã trên bốn ngôn ngữ. Đây không phải 1.600 ca nghe. Có thêm 600 số VI/EN/KO đối chiếu với num2words sau bỏ khác biệt cách nối. [N1, N2]')
item('10 nhóm unittest:', 'Đều qua; có ngày/giờ không hợp lệ, ngày mơ hồ, ranh giới từ và giới hạn đầu vào Hangul sau NFKD.')
item('50 ca en-vi:', '49 bản có text tạo audio TTS thành công; EV-2748 rỗng nên bỏ qua. Cả 49 đầu vào TTS khớp formatter cuối cùng. Không thay file gốc.')
item('Chạy thực:', '4 câu thử TTS, mỗi ngôn ngữ một câu; 6 ca S2S gồm 3 audio EN gốc và cùng audio VI sang EN/KO/ZH. Không truyền transcript vào ASR. Tổng 59 WAV mới đã kiểm dữ liệu hữu hạn, có mẫu. [N1, N3]')
p('Trong lần S2S Hàn đầu tiên, guard đã chặn đầu vào vượt giới hạn NFKD. Sau khi sửa cách tính ngân sách và chạy lại, pipeline hoàn tất. Một test khác xác nhận ca rỗng không tái sử dụng transcript TTS của ca trước.','small')
h('5. Audio đã tốt hoàn toàn chưa?')
p('<b>Chưa thể kết luận.</b> Whisper Small cục bộ được dùng nhận dạng lại 7 audio mới, không có prompt tham chiếu. Đây là bằng chứng đối chiếu, không phải người nghe hoặc MOS. [N4]','callout')
item('Ba ca EN→VI:', 'ASR nghe lại khôi phục được 5 giờ 30 phút, 8 giờ, 734 và 67297. Từ taxi/tên địa chỉ vẫn có sai khác.')
item('Câu tổng hợp Anh:', 'Giờ, ngày, tiền, nhiệt độ, ZIP, điện thoại khớp khi nhận dạng lại; VN bị chép thành VIN.')
item('Câu tổng hợp Việt/Hàn/Trung:', 'Còn sai khác khi nhận dạng lại ở tiền, ngày/giờ, số nhà hoặc mã dù văn bản đưa vào TTS đúng. Cần nghe để phân biệt lỗi TTS thực với lỗi ASR đối chiếu.')
h('6. Những gì format chưa giải quyết')
p('ASR/MT vẫn có thể sai trước TTS: 32→2, trưa→tối, “báo động” thay “báo thức”, hoặc dịch sai thành ngữ. Audio VI thử S2S còn mất/sai giờ, ngày, tiền và nhiệt độ từ ASR. Formatter không tự khôi phục giá trị nguồn đã sai.')
p('Chưa bao phủ mọi dạng văn bản: 03/04 cần chọn DMY/MDY; dấu thập phân theo locale; $ hiện quy ước USD; H:MM chưa phân biệt mọi tỷ lệ/thời lượng. Ký hiệu khoa học và khoảng số được cảnh báo, chưa đọc đầy đủ theo nghĩa. Từ đơn quá dài vượt graph sẽ báo lỗi.','small')

new()
p('Ưu tiên công việc tiếp theo', 'title')
p('Đây là đề xuất, chưa triển khai. Mỗi bước cần tiêu chí nghiệm thu riêng.', 'subtitle')
h('P0 - Chốt chất lượng audio trước khi mở rộng')
item('Làm gì:', 'Nhờ người thông thạo từng ngôn ngữ nghe audio có sai khác, ghi nguyên văn phần nghe được và đánh dấu lỗi số/đơn vị/mã. Đối chiếu từng bước: MT → tts_text → chunk → audio.')
item('Đạt khi:', 'Tách được lỗi chuẩn hóa, chia đoạn, G2P/TTS và ASR nghe lại; các ca số/mã quan trọng được xác nhận đúng. Không sửa formatter chỉ để khớp một giả thuyết Whisper.')
h('P0 - Bảo vệ giá trị số xuyên suốt S2S')
item('Làm gì:', 'Lưu chuỗi số/mã có kiểu từ ASR, theo dõi thay đổi qua MT và TTS; tăng kiểm tra số 0, dấu âm, phần thập phân, đơn vị và số chữ số. Với giá trị mơ hồ, báo cờ thay vì tự đoán.')
item('Đạt khi:', 'Phát hiện được nơi làm đổi 32→2 hoặc mất chữ số; đầu vào TTS không âm thầm thay đổi giá trị đã được xác nhận. Kiểm tra kiểu dữ liệu không thay thế kiểm chứng lời đọc thực.')
h('P1 - Mở rộng ngữ cảnh và dữ liệu kiểm thử độc lập')
item('Làm gì:', 'Phân biệt giờ/tỷ lệ/thời lượng, mã/số lượng, định dạng ngày và tiền tệ theo locale. Bổ sung tagger đa ngôn ngữ hoặc dữ liệu phù hợp Hàn/Trung; người bản ngữ kiểm tra bảng tên chữ cái.')
item('Đạt khi:', 'Cùng một bộ xử lý chạy được trên tập câu mới chưa dùng phát triển, gồm ngữ cảnh gây nhầm và mã có dấu. Báo riêng tỷ lệ nhận đúng loại, giữ đúng giá trị và nghe đúng mã.')
h('P2 - Đo và vận hành trên hệ thống đích')
item('Làm gì:', 'Đo cold/warm latency, p50/p95, RAM và thời gian ra audio trên thiết bị đích; test câu dài và nhiều yêu cầu. Gắn phiên bản model, locale, cấu hình format và hash đầu vào vào mỗi kết quả.')
item('Đạt khi:', 'Có ngưỡng latency/chất lượng được thống nhất; truy vết và tái tạo được lỗi; có regression test và cách quay về legacy. Kết quả hiện tại là CPU Mac, chưa chứng minh hiệu năng NPU.')
h('Nguồn và nơi xem bằng chứng')
p('<b>Thư mục gốc:</b> OneVoice_integrated/ (trong workspace hiện tại). Các nguồn nội bộ dưới đây là file thực nghiệm, không phải paper.','small')
p('<b>[N1]</b> results/hybrid_format/evaluation.json: số đo, kiểm tra WAV, hash và kết quả HTTP.<br/><b>[N2]</b> results/hybrid_format/semantic_examples.json; cardinal_crosscheck.json; tests/test_hybrid_format.py.<br/><b>[N3]</b> results/en_vi_format_replay/hybrid/; results/multilingual_tts_format/hybrid/; results/hybrid_format/s2s/.<br/><b>[N4]</b> results/hybrid_format/audio_crosscheck.json: nguyên văn 7 giả thuyết ASR.<br/><b>Bản chi tiết và liên kết audio:</b> results/hybrid_format/README.md.','small')
p('Tham khảo kiến trúc (tài liệu web, không có số trang): <link href="https://github.com/NVIDIA/NeMo-text-processing/blob/main/nemo_text_processing/hybrid/README.md" color="#087F84">NVIDIA NeMo Hybrid TN</link>; <link href="https://unicode-org.github.io/icu/userguide/format_parse/numbers/rbnf.html" color="#087F84">ICU RuleBasedNumberFormat</link>. Triển khai hiện tại dùng grammar/codec riêng và CRF sẵn có, không phải bản tích hợp đầy đủ NeMo/ICU.','small')

def decorate(c, doc):
    w, h = A4
    c.setFillColor(TEAL); c.rect(0, h-9, w, 9, fill=1, stroke=0)
    c.setStrokeColor(HexColor('#D6E1E5')); c.line(42, 36, w-42, 36)
    c.setFont('Body', 7.5); c.setFillColor(GRAY)
    c.drawString(42, 23, 'ONEVOICE | TTS FORMAT | 12/11/2026')
    c.drawRightString(w-42, 23, f'{doc.page} / 3')

target = ROOT/'output/pdf/bao-cao-format-tts-hybrid.pdf'
target.parent.mkdir(parents=True, exist_ok=True)
doc = SimpleDocTemplate(str(target), pagesize=A4, rightMargin=42, leftMargin=42,
                        topMargin=35, bottomMargin=47, title='Format TTS hybrid - Kết quả và kế hoạch tiếp theo',
                        author='OneVoice project')
doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
print(target)
