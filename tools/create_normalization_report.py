from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,KeepTogether
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT
r=Path(__file__).resolve().parents[1]
f=Path('/Users/khoa/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/libreoffice-headless/libreoffice/LibreOfficeDev.app/Contents/Resources/fonts/truetype')
pdfmetrics.registerFont(TTFont('DV',str(f/'DejaVuSans.ttf')))
pdfmetrics.registerFont(TTFont('DV-B',str(f/'DejaVuSans-Bold.ttf')))
pdfmetrics.registerFontFamily('DV',normal='DV',bold='DV-B',italic='DV',boldItalic='DV-B')
styles={
 'title':ParagraphStyle('title',fontName='DV-B',fontSize=22,leading=28,textColor=HexColor('#11354b'),spaceAfter=12),
 'sub':ParagraphStyle('sub',fontName='DV',fontSize=9,leading=14,textColor=HexColor('#506572'),spaceAfter=13),
 'h':ParagraphStyle('h',fontName='DV-B',fontSize=12,leading=17,textColor=HexColor('#126e78'),spaceBefore=12,spaceAfter=5),
 'body':ParagraphStyle('body',fontName='DV',fontSize=9.5,leading=14.4,spaceAfter=7),
 'small':ParagraphStyle('small',fontName='DV',fontSize=8.3,leading=12.2,spaceAfter=7),
 'quote':ParagraphStyle('quote',fontName='DV',fontSize=8.5,leading=12.5,textColor=HexColor('#445b65'),leftIndent=12,spaceAfter=8),
}
story=[]
def p(t,kind='body'):story.append(Paragraph(t,styles[kind]))
def h(t):p(t,'h')
def page():story.append(PageBreak())
p('Chuẩn hóa Anh - Việt<br/>Sửa lỗi bằng mô hình học từ dữ liệu','title')
p('OneVoice S2S | Báo cáo nghiên cứu ngắn | 12/11/2026 | Phạm vi: EN và VI','sub')
p('<b>Kết luận:</b> Có thể xử lý giờ như <b>3:30</b> mà không viết lệnh thay thế cho từng giá trị. Hướng phù hợp là mô hình học ngữ cảnh để nhận diện đoạn cần đổi, rồi mô hình sinh cách đọc. Không có bảo đảm đúng mọi văn bản; đặc biệt text <b>330</b> đơn lẻ không đủ xác định đó là giờ hay số đếm.')
h('1. Phân biệt đúng lỗi trước khi sửa')
p('<b>ITN sau ASR:</b> dạng nói → dạng viết, ví dụ “three thirty p m” → “3:30 PM”. <b>TN trước TTS:</b> dạng viết → cách đọc, ví dụ “3:30 chiều” → “ba giờ ba mươi chiều”. Hai chiều xử lý khác nhau; mô hình ITN không tự thay được TN. [1, tr. PDF 1-2; 2, tr. PDF 1-3]')
p('<b>Bằng chứng từ lần chạy hiện tại:</b> ASR EN trả “330 pm”; NLLB dịch thành “330 giờ tối”. ASR VI nhận sai “năm mươi nghìn” thành “năm nghìn”; viBERT-CAPU chèn dấu câu vào chuỗi số. Đây là lỗi trước TTS, không thể chỉ đổi hàm đọc số để sửa hết. [L1]')
h('2. Trường hợp 3:30: giải pháp cụ thể')
p('<b>Khi text còn đúng “3:30” / “14h45”:</b> huấn luyện TN theo câu để học lớp TIME và sinh “ba giờ ba mươi” / “mười bốn giờ bốn mươi lăm”. Dữ liệu gồm nhiều giờ, cách viết và ngữ cảnh, không gán sẵn một câu đọc cho mỗi thời điểm. Giữ rõ “sáng/chiều” nếu đầu vào có; không tự thêm khi thiếu. [2, tr. PDF 2-3]')
p('<b>Khi ASR đã trả “330 pm”:</b> mô hình ITN/chỉnh sửa học cách khôi phục dấu hai chấm dựa vào ngữ cảnh. Nếu có nguy cơ nhầm với số đếm, chạy lượt hai có nghe lại audio và xét giả thuyết ASR. Đây là đề xuất áp dụng từ nghiên cứu, chưa phải khả năng sẵn có của graph hiện tại. [1, tr. 551; 3, tr. PDF 1-2]')
p('<b>Không nên:</b> thay mọi “330” thành “3:30”. “Phòng 330”, “giá 330 đồng” và “3:30 chiều” phải cho ba cách đọc khác nhau. Tập kiểm thử phải có cả các cặp gây nhầm này.')
h('3. “Không code cứng” nghĩa là gì ở đây?')
p('Không dùng regex/WFST viết tay để quyết định cách đọc từng loại. Quyết định nhận diện và biến đổi do trọng số học từ các cặp ví dụ. Vẫn cần mã điều phối, chuẩn dữ liệu và chính sách đọc mã từng ký tự. Ràng buộc giữ nguyên chuỗi số là một lựa chọn kỹ thuật riêng; không nên gọi nó là hệ thống hoàn toàn không có quy tắc.')
page()
p('Từng nhóm lỗi và cách sửa','title')
p('“Đã thấy” = xuất hiện trong log/docs; “cần kiểm” = rủi ro suy từ mã, chưa xác nhận đầy đủ bằng nghe. Các cách sửa dưới đây là đề xuất, chưa triển khai.','sub')
items=[
('01 / Giờ, AM-PM - đã thấy','3:30 → “ba : ba mươi”; 14h45 giữ chữ h; ASR cho 330 pm.','Học TIME theo cả câu bằng tagger + decoder; ITN sau ASR khi cần khôi phục định dạng. Nghe lại audio khi số đã bị nhận sai. [1, tr. 551; 2, tr. PDF 2-3; 3, tr. PDF 1-2]'),
('02 / Ngày, tháng, năm - đã thấy','15/11/2026 bị đọc thành các số rời, thiếu “ngày/tháng/năm”.','TN học cặp ngày viết - ngày đọc; thêm nhãn ngôn ngữ/locale cho 03/04. Dữ liệu phải có ngày, phân số và mã dạng slash để học phân biệt. Năm đứng riêng cũng cần ngữ cảnh. [2, tr. PDF 1-3]'),
('03 / Thập phân và phân cách hàng nghìn - đã thấy','37.5 → 37. 5 ở MT; 1.234 bị mặc định là 1234 ở TN.','Học phân đoạn số theo ngữ cảnh và locale, không xóa dấu trước khi mô hình phân loại. Fine-tune MT để giữ giá trị; học TN từ các biến thể có khoảng trắng do ASR/MT tạo ra. [2, tr. PDF 2-3; 4, tr. 1537-1538]'),
('04 / Tiền, phần trăm, đơn vị - một phần đã thấy','$15.50 → “15 đô la. 50”; 50.000 đồng và 15% đã qua ví dụ đơn giản.','Học MONEY / PERCENT / MEASURE cùng cách đọc: tiền có đơn vị chính/phụ; °C phải có tên đơn vị. Huấn luyện trên cả câu dịch và câu gốc. Không suy từ một ví dụ đúng rằng mọi loại tiền đều đúng. [2, tr. PDF 1-3]'),
('05 / Điện thoại, ZIP, mã chữ-số - đã thấy','Mất số 0 đầu; VN123 và 67297 bị đọc như số lượng.','Mô hình nhận diện span + decoder ở mức ký tự, có cơ chế sao chép. Huấn luyện với nhãn PHONE / CODE và yêu cầu đọc từng ký tự; giữ toàn bộ chữ số gốc, kể cả 0. Chuỗi không có ngữ cảnh vẫn có thể mơ hồ. [2, tr. PDF 2-3]'),
('06 / Tên ngoại ngữ, viết tắt, từ gạch nối - docs ghi lỗi','Johnson, Main Street, check-out có thể bị đọc sai hoặc bỏ.','Dùng G2P học từ dữ liệu hoặc TTS đa ngữ. G2P tạo âm vị, chưa tự là phiên âm Việt; Piper fused hiện dùng bảng từ nên cần đổi frontend/graph để nhận âm vị mới. Không chỉ sửa textnorm. [5, tr. 3330-3332; L2]'),
]
for title,err,fix in items:
 story.append(KeepTogether([Paragraph(title,styles['h']),Paragraph('<b>Lỗi:</b> '+err,styles['body']),Paragraph('<b>Cách sửa:</b> '+fix,styles['body'])]))
p('<b>Bổ sung cần kiểm:</b> số âm, phân số, số thứ tự, email/URL, phiên bản và số phòng cần có lớp/cặp dữ liệu riêng. Nếu chữ đầu vào đúng nhưng âm thanh sai thanh điệu, đó là lỗi phát âm/TTS, không phải lỗi TN. [L2]','small')
page()
p('Ba hướng khả thi, không vá từng câu','title')
p('Đánh giá latency và dung lượng dưới đây là định tính theo kiến trúc, chưa phải benchmark trên OneVoice.','sub')
h('A / Tagger theo ngữ cảnh + decoder nhỏ - ưu tiên')
p('Encoder học chọn span và loại dữ liệu; decoder chỉ sinh cách đọc cho span cần đổi, giữ phần còn lại. Dùng cặp EN/VI dạng viết - dạng nói; huấn luyện thêm dữ liệu nhiễu từ ASR/MT thật. Thutmose là tham khảo cho <b>ITN</b> gán nhãn; mô hình TN hai giai đoạn của Ro và cộng sự phù hợp cho chiều trước TTS. [1, tr. 551-552; 2, tr. PDF 2-3]')
p('<b>Đánh đổi:</b> thường giảm lượng text phải sinh so với viết lại cả câu, nhưng decoder vẫn tự hồi quy. Cần dữ liệu Việt; nhãn rời rạc có thể hạn chế dạng chưa thấy. Paper Thutmose thử EN/RU và có bước xử lý dữ liệu bằng regex; không lấy nguyên hệ thống đó rồi tuyên bố hoàn toàn không có quy tắc.','small')
h('B / Seq2seq toàn câu hoặc mô hình sinh các phép sửa')
p('Một Transformer học trực tiếp câu viết → câu đọc, hoặc học vị trí và nội dung cần thay. Dễ hỗ trợ định dạng mới bằng dữ liệu thay vì sửa grammar. Bản sinh phép sửa có thể giữ nguyên phần văn bản không cần đổi. [2, tr. PDF 2, mục 2.1]')
p('<b>Đánh đổi:</b> có thể thay nhầm số/đơn vị, bỏ từ; khó bảo đảm giữ mã chính xác. Với cùng quy mô và chiều dài, sinh cả câu thường tốn thêm bước giải mã; cần đo thực tế. Không chọn LLM lớn chỉ để xử lý vài span số.','small')
h('C / Lượt hai nghe audio + text - dành cho lỗi ASR')
p('Dùng encoder âm thanh và giả thuyết ASR để giải mã lại đoạn có số/giờ chưa chắc. Có cơ sở để sửa chỗ text đã làm mất thông tin; mô hình chỉ nhìn text không có lợi thế này. [3, tr. PDF 1-2]')
p('<b>Đánh đổi:</b> tích hợp khó và thêm bộ nhớ/độ trễ; graph ASR hiện tại không xuất đủ đặc trưng/giả thuyết để gắn ngay. Chỉ chạy trên đoạn nghi ngờ là đề xuất tối ưu, chưa được chứng minh cho repo. Đây là bổ sung cho A/B, không thay TN.','small')
h('Giữ số qua MT và dấu câu: cần sửa cùng pipeline')
p('Fine-tune bộ dấu câu trên transcript có số thật, hoặc học chung nhãn span số và dấu câu; JointCapPunc là bằng chứng cho hướng học chung dấu câu/viết hoa, <b>không chứng minh đã bảo vệ số</b>. Với NLLB, thử fine-tune giữ số hoặc constrained decoding theo span do mô hình phát hiện. Đây là hướng áp dụng từ [4, tr. 1537-1538] và [6, tr. PDF 2-3]; chưa được kiểm chứng trên NLLB pruned của dự án.')
h('Cách chọn và kiểm chứng')
p('Bắt đầu với A cho TN; bổ sung ITN và C khi cần sửa ASR. Chia tập test theo giá trị chưa thấy, kiểu viết, người nói và miền nội dung. Đo riêng: số/chữ số có bị đổi hoặc mất không, độ đúng cách đọc theo loại, p50/p95 latency và RAM trên thiết bị đích. Nghe mù các ca 3:30, 330, 37.5, 37. 5, 03/04, số có 0 đầu. Không dùng WER tổng làm thước đo duy nhất.')
page()
p('Nguồn và phạm vi bằng chứng','title')
p('Số trang PDF tính từ 1. Trang in của hội nghị được ghi riêng nếu dùng. Các ví dụ tiếng Việt trong báo cáo là mục tiêu thiết kế, không phải kết quả đo từ paper.','sub')
refs=[
('[1] Antonova, Bakhturina & Ginsburg (2022). Thutmose Tagger: Single-pass neural model for Inverse Text Normalization.','https://www.isca-archive.org/interspeech_2022/antonova22_interspeech.pdf','Tr. in 550-554; kiến trúc và alignment: 551-552 (PDF 2-3).','“The model outputs tags for all input words.”'),
('[2] Ro, Stahlberg, Wu & Kumar (2022). Transformer-based Models of Text Normalization for Speech Applications.','https://arxiv.org/pdf/2202.00153','Tr. PDF 1-3; mục 2.1: phép sửa; mục 2.2: mô hình hai giai đoạn.','“jointly segments and coarsely classifies tokens”'),
('[3] Hu, Sainath, Pang & Prabhavalkar (2020). Deliberation Model Based Two-Pass End-to-End Speech Recognition.','https://arxiv.org/pdf/2003.07962','Tr. PDF 1-2, mục 2.1: kết hợp audio và giả thuyết ASR.','“attend to both acoustics and first-pass hypotheses”'),
('[4] Hokamp & Liu (2017). Lexically Constrained Decoding for Sequence Generation Using Grid Beam Search.','https://aclanthology.org/P17-1141.pdf','Tr. in 1537-1538 (PDF 3-4), mục 3. Giữ chuỗi bắt buộc, không tự bảo đảm đúng nghĩa số.',''),
('[5] Yao & Zweig (2015). Sequence-to-Sequence Neural Net Models for Grapheme-to-Phoneme Conversion.','https://www.isca-archive.org/interspeech_2015/yao15_interspeech.pdf','Tr. in 3330-3332 (PDF 1-3): bài toán chữ → âm vị và mô hình học. Không chứng minh phiên âm Việt.',''),
('[6] Hoang Thi Thu Uyen, Nguyen Anh Tu & Ta Duc Huy (2022). Vietnamese Capitalization and Punctuation Recovery Models.','https://arxiv.org/pdf/2207.01312','Tr. PDF 2-3, mục 4-5: JointCapPunc và thí nghiệm trên ViCapPunc, thiên về miền y tế.',''),
]
for title,url,loc,quote in refs:
 p(title,'small');p('<link href="'+url+'" color="#126e78">'+url+'</link><br/>'+loc,'small')
 if quote:p('Trích ngắn: '+escape(quote),'quote')
h('Nguồn nội bộ đã đối chiếu')
p('[L1] results/rerun_2026-11-12/{en_to_vi,vi_to_en}.json và transcript.txt: ASR thô, sau dấu câu, bản dịch; lần chạy 12/11/2026. Đây là hai fixture tổng hợp, không phải benchmark chất lượng.','small')
p('[L2] s2s_edge/docs/tts_format_issues.md, mục 1, 4 và 6; s2s_edge/s2s_edge/textnorm.py; tts.py. Tài liệu cũ ghi EN “chưa thấy lỗi” nhưng cũng ghi chưa kiểm giờ, tiền, điện thoại. Do đó không kết luận chuẩn hóa EN đã hoàn tất.','small')
p('<b>Trạng thái:</b> báo cáo nghiên cứu; chưa sửa code hoặc train model mới. Không dùng kết quả EN/RU trong paper làm cam kết chất lượng/latency VI. Không có phương án học máy nào được chứng minh đúng mọi text.','small')

def footer(c,doc):
 c.setStrokeColor(HexColor('#d8e2e7'));c.line(43,40,552,40);c.setFont('DV',8);c.setFillColor(HexColor('#506572'));c.drawString(43,26,'ONEVOICE / NGHIÊN CỨU CHUẨN HÓA ANH - VIỆT');c.drawRightString(552,26,str(doc.page))
path=r/'output/pdf/bao-cao-fix-normalize-anh-viet.pdf'
doc=SimpleDocTemplate(str(path),pagesize=(595.28,841.89),rightMargin=43,leftMargin=43,topMargin=40,bottomMargin=54,title='OneVoice - Sửa lỗi chuẩn hóa Anh Việt bằng mô hình học',author='OneVoice research')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(path)
