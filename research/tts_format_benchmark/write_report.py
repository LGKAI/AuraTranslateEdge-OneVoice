"""Vietnamese PDF report assembled strictly from measured rerun artifacts."""
from pathlib import Path
import json,re,statistics,html,hashlib
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from reportlab.graphics.shapes import Drawing,Rect,String
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'output/tts-format-rerun-2026-11-11'
OUT=ROOT/'output/pdf/bao-cao-tts-anh-viet-3-phuong-phap.pdf';OUT.parent.mkdir(parents=True,exist_ok=True)
R=json.loads((DATA/'results.json').read_text());E=R['environment'];S=R['summary']
OLD=json.loads((DATA/'before_identifier_policy/results.json').read_text())
INV=json.loads((DATA/'branch_model_inventory.json').read_text())
F='/System/Library/Fonts/Supplemental/'
pdfmetrics.registerFont(TTFont('Arial',F+'Arial.ttf'))
pdfmetrics.registerFont(TTFont('ArialB',F+'Arial Bold.ttf'))
pdfmetrics.registerFont(TTFont('ArialI',F+'Arial Italic.ttf'))
pdfmetrics.registerFontFamily('Arial',normal='Arial',bold='ArialB',italic='ArialI',boldItalic='ArialB')
INK=colors.HexColor('#182832');MUTED=colors.HexColor('#53616b');ACCENT=colors.HexColor('#176b75');LIGHT=colors.HexColor('#eef4f5');LINE=colors.HexColor('#d5dfe3')
W,H=A4;MARGIN=44;WIDTH=W-2*MARGIN
ST={
 'title':ParagraphStyle('title',fontName='ArialB',fontSize=25,leading=31,textColor=INK,spaceAfter=15),
 'h1':ParagraphStyle('h1',fontName='ArialB',fontSize=18,leading=24,textColor=INK,spaceAfter=12),
 'h2':ParagraphStyle('h2',fontName='ArialB',fontSize=12.2,leading=16,textColor=ACCENT,spaceBefore=10,spaceAfter=6),
 'body':ParagraphStyle('body',fontName='Arial',fontSize=10.1,leading=14.5,textColor=INK,spaceAfter=8),
 'small':ParagraphStyle('small',fontName='Arial',fontSize=8.4,leading=11.3,textColor=MUTED,spaceAfter=6),
 'cell':ParagraphStyle('cell',fontName='Arial',fontSize=8.5,leading=11.3,textColor=INK),
 'tiny':ParagraphStyle('tiny',fontName='Arial',fontSize=7.7,leading=10.1,textColor=INK),
 'quote':ParagraphStyle('quote',fontName='ArialI',fontSize=9.6,leading=13.3,textColor=MUTED,leftIndent=12,borderColor=LINE,borderWidth=.6,borderPadding=8,spaceBefore=5,spaceAfter=9),
 'transcript':ParagraphStyle('transcript',fontName='Arial',fontSize=9.1,leading=12.3,textColor=INK,spaceAfter=8),
}
flow=[]
def clean(s):return str(s).replace('\u2011','-').replace('\u2013','-').replace('\u2014','-').replace('\u00a0',' ')
def p(s,style='body',raw=False):return Paragraph(clean(s) if raw else html.escape(clean(s)),ST[style])
def add(s,style='body',raw=False):flow.append(p(s,style,raw))
def heading(n,title):add(f'{n}. {title}','h1')
def page():flow.append(PageBreak())
def tbl(rows,widths=None,small=False):
 st='tiny' if small else 'cell'
 data=[[p(str(x),st,raw=False) for x in row] for row in rows]
 t=Table(data,colWidths=widths or [WIDTH/len(rows[0])]*len(rows[0]),repeatRows=1,hAlign='LEFT')
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),LIGHT),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.8,ACCENT),('LINEBELOW',(0,1),(-1,-1),.35,LINE),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6)]))
 flow.append(t);flow.append(Spacer(1,9));return t

def metric(v):return f"{v['passed']}/{v['total']} ({100*v['passed']/v['total']:.1f}%)"
def fnum(x):return f'{x:.2f}'
def label(k):return {'B0':'B0 - TN hiện tại + bảo vệ mã','M1':'M1 - WFST','M2':'M2 - CRF + WFST','M3':'M3 - WFST + DistilBERT'}[k]
def cell(v):return f"{v['passed']}/{v['total']}"
def quote(text,source):add('“'+text+'”','quote');add(source,'small')

def chart():
 d=Drawing(WIDTH,135)
 for i,m in enumerate(['B0','M1','M2','M3']):
  y=108-i*27;v=100*S[m]['passed']/96
  d.add(String(0,y+3,m,fontName='ArialB',fontSize=9,fillColor=INK))
  d.add(Rect(35,y,WIDTH-95,15,fillColor=LIGHT,strokeColor=None))
  d.add(Rect(35,y,(WIDTH-95)*v/100,15,fillColor=ACCENT if m!='B0' else MUTED,strokeColor=None))
  d.add(String(WIDTH-52,y+3,f'{v:.1f}%',fontName='ArialB',fontSize=9,fillColor=INK))
 return d

# 1 - cover and measured decision.
add('ONEVOICE / KIỂM THỬ LẠI','small')
add('Chuẩn hóa TTS\nAnh - Việt'.replace('\n','<br/>'),'title',True)
add('So sánh 3 phương pháp sau khi áp dụng cách đọc từng ký tự cho điện thoại và mã định danh.','body')
add('Ngày 11/11/2026 · contest_VNG '+E['branches']['contest_VNG'][:7]+' + thay đổi cục bộ chưa commit.','small')
add('Kết quả chính','h2')
add('WFST và CRF + WFST cùng đạt 76/96 ca theo kỳ vọng văn bản (79,2%). WFST + DistilBERT đạt 72/96 (75,0%). Cả ba đều đạt 16/16 ca điện thoại/mã sau khi dùng lớp bảo vệ định danh chung.')
flow.append(chart())
add('Tỷ lệ đạt trên 96 ca tự soạn, không phải WER/MOS của audio. Nguồn: kết quả chạy mới tại results.json, '+E['finished_local'][:19].replace('T',' ')+'.','small')
tbl([['Phương pháp','Đạt / 96','p50 TN (ms)','p95 TN (ms)']]+[[m,metric(S[m]),fnum(S[m]['p50_ms']),fnum(S[m]['p95_ms'])] for m in ['M1','M2','M3']],[95,180,WIDTH/2-137.5,WIDTH/2-137.5])
add('Khuyến nghị hiện tại: dùng M1 làm nền cho thiết bị; giữ M2 để cải thiện ngữ cảnh khi có dữ liệu thật. M3 hiện chậm hơn rõ rệt và chưa tăng tỷ lệ đạt trên bộ ca này. Đây là kết luận của cấu hình thử nghiệm, không phủ định kiến trúc trong paper.')
add('Đã chạy lại 96 ca nhỏ và 2 đoạn tổng hợp; xuất 6 audio so sánh + 2 audio B0. Latency được đo trên CPU máy Mac; không phải số đo QCS8550/NPU.','small')
page()
# 2 - current code and inventory.
heading(1,'Hệ thống thực tế và model hiện có')
add('Bản làm mốc là thư mục AuraTranslateEdge_branches/contest_VNG. Bản nguồn AuraTranslateEdge được đồng bộ cùng thay đổi ở identifiers.py, tts.py và tests_identifiers.py; các nhánh chuyên biệt khác giữ nguyên.')
add('Đã quét toàn bộ file trong 8 thư mục nhánh, gồm file bị gitignore ẩn, loại riêng thư mục metadata .git. Ảnh chụp trước sửa có số liệu sau:','small')
tbl([['Nhánh','Commit','Số file','Dung lượng nội dung']]+[[v['branch'],v['commit'][:7],str(v['file_count']),f"{v['total_bytes']/1e6:.2f} MB"] for v in INV],[105,90,75,WIDTH-270])
add('Có trong các nhánh: mã nguồn/export model, cấu hình, tokenizer, bảng phụ trợ, log và audio kết quả. Trọng số ONNX có sẵn là GTCRN (khử nhiễu); contest_VNG còn có vector giọng Supertonic. Không tìm thấy trọng số TTS gộp Piper/Supertonic, NLLB hoặc ASR lớn trong 8 working tree. Không có file con trỏ Git LFS trong ảnh chụp.')
add('Bằng chứng trực tiếp: piper_npu/.gitignore loại build/, data/, outputs/, *.onnx; supertonic_npu/.gitignore loại work/ và *.onnx. models.dev_beam.json đang trỏ đường dẫn Windows C:/Users/Admin/Downloads/...; các file đó không hiện diện trên máy này. [L1-L3]','small')
add('Model dùng cho lần nghe thử','h2')
add('Tiếng Việt: Piper vi_VN-vais1000-medium gốc đã tải (~63 MB). Tiếng Anh: 4 ONNX Supertonic 3 gốc có sẵn trong ~/.cache/supertonic3. Dùng cùng giọng/cài đặt giữa các phương pháp. Chưa tái tạo graph gộp, chưa chạy board, chưa chạy ASR/MT đầu-cuối.')
add('Danh mục file/byte và các quy tắc gitignore được lưu trong branch_model_inventory.json để đối chiếu. Model trong cache không đồng nghĩa với model gộp/NPU có trong Git clone.','small')
page()
# 3 - identifier correction.
heading(2,'Chính sách đọc điện thoại và mã')
add('Nhận diện định danh ngay trên văn bản gốc, trước khi phiên âm từ Latin và chuyển chữ thường. Sau đó dùng một bộ chuyển ký tự dùng chung cho B0, M1, M2 và M3: mỗi chữ/số tạo một đơn vị đọc; giữ số 0 đầu và dấu cộng quốc tế.')
tbl([['Đầu vào','Cách đọc yêu cầu'],['VN123','vê en một hai ba / v n one two three'],['AB-0012','a bê không không một hai'],['ZIP code 67297','ZIP code six seven two nine seven'],['0912 345 678','không chín một hai ba bốn năm sáu bảy tám'],['+84 912 345 678','cộng tám bốn chín một hai ba bốn năm sáu bảy tám']],[145,WIDTH-145])
add('Cách mở rộng','h2')
add('Một parser trả về span kiểu telephone/code dựa vào hình thức và tín hiệu ngữ cảnh như điện thoại, gọi, ZIP code, mã bưu chính, PIN, chuyến bay. Bộ verbalizer dùng bảng chữ cái/chữ số cho mọi giá trị trong span; không viết phép thay thế riêng cho từng mã.')
add('Số lượng bình thường vẫn dùng ngữ pháp số: “Có 67297 người” đọc theo số lượng; “mã bưu chính 67297” đọc từng chữ số. Các kiểu giờ 14h45, đơn vị 10km, số thứ tự 21st, khoa học 1e-3 và email/URL được tránh nhận nhầm thành mã.')
add('Trong chuỗi xử lý hiện tại','h2')
add('Văn bản gốc → nhận diện & đọc định danh → phiên âm phần còn lại (VI) → chuẩn hóa bằng phương pháp được chọn → chia đoạn → front-end TTS → model âm học.')
add('5 nhóm unit test với các subcase đã qua: đọc chữ/số; nhận diện điện thoại/mã; tránh nhầm số lượng/giờ/đơn vị; bảo vệ khỏi phiên âm tên; giữ nguyên giá trị khi đổi normalizer. Trong 96 ca, cả 8 điện thoại và 8 mã đều đạt với từng phương pháp.')
add('Điểm tăng ở điện thoại/mã là hiệu quả của lớp bảo vệ chung theo yêu cầu mới; không phải bằng chứng CRF hoặc DistilBERT tự học cách đọc mã. Giới hạn: parser vẫn cần quy ước kiểu/ngữ cảnh và chưa bao phủ mọi mã hay định dạng quốc tế.','small')
page()
# 4 - methods and source excerpts.
heading(3,'Ba phương pháp được thử')
add('M1 - WFST ưu tiên cố định','h2')
quote('Text normalization (TN) systems in production are largely rule-based using weighted finite-state transducers (WFST).','[S1] Abstract, PDF trang 1 / kỷ yếu trang 491.')
add('Ý tưởng: dùng ngữ pháp tổ hợp để tạo cách đọc. Bản thử nghiệm dùng NeMo/Pynini cho từng span, bổ sung graph đánh vần chung và chọn ứng viên ưu tiên đầu. Có xử lý theo lớp, nhưng vẫn cần grammar; đây không phải toàn bộ normalizer NeMo chạy nguyên câu.')
add('M2 - CRF chọn lớp + WFST chuyển cách đọc','h2')
quote('First, a model-based tagger is designed to detect NSWs.','[S2] Abstract, PDF trang 1; kiến trúc tại mục III, PDF trang 2-3. Dấu nối/ngắt dòng đã chuẩn hóa.')
add('Ý tưởng: dùng ngữ cảnh để gán lớp trước khi mở rộng. Bản thử nghiệm dùng CRF đóng băng đã huấn luyện trên 2.304 câu từ template tổng hợp, seed 521; đặc trưng hình dạng token và cửa sổ ±4 token. Chọn ứng viên đầu cùng lớp, nếu không có thì quay về ưu tiên WFST. Không huấn luyện lại trên 96 nhãn test, không tái lập mô hình/dataset của paper.')
add('M3 - WFST + DistilBERT đa ngôn ngữ','h2')
quote('a neural language model picks the best one','[S1] Abstract, PDF trang 1; cơ chế WFST + LM: mục 2, PDF trang 2-3 / kỷ yếu trang 492-493.')
add('Bản thử nghiệm dùng DistilBERT multilingual cased, 135,45 triệu tham số, không fine-tune. Chấm pseudo-log-likelihood cho các token thay thế trong ngữ cảnh; loại ứng viên còn chữ số nếu có ứng viên đã chuyển sang chữ. Giới hạn 128 token, ngữ cảnh theo câu, 4 luồng CPU. Đây là tái triển khai giản lược: không có toàn bộ trọng số/pruning/benchmark của paper. [S4]')
add('Điểm chung','h2')
add('B0 là normalizer cơ bản đang được tts.py gọi. M1/M2/M3 được ghép vào cùng vị trí trong harness kiểm thử; chưa có bộ chọn ba phương pháp trong runtime production. Cả ba dùng cùng lớp định danh, cùng tập ứng viên grammar và cùng dữ liệu thử. Tên nước ngoài chưa có G2P mới; đang dùng bộ phiên âm cũ. Cơ chế WFST chỉ bảo vệ được điều grammar thực sự bao phủ; mô hình chọn sai lớp/cách đọc vẫn có thể làm đổi ý nghĩa.')
page()
# 5 - protocol.
heading(4,'Thiết kế test và cách chấm')
tbl([['Mức độ','Số ca','Nội dung'],['1 - Dễ','24 (12 VI + 12 EN)','Số nguyên, số thập phân thường, tiền, %, số phòng/năm, giữ câu thường'],['2 - Định dạng','24','Giờ, ngày, điện thoại, mã, số âm, đơn vị, ordinal, phân số, khoảng số'],['3 - Ngữ cảnh','24','1/4 ngày hay phân số; số lượng hay ZIP; PIN có số 0; phiên bản; tiền nhiều dấu phân cách'],['4 - Đặc biệt','24','Điện thoại quốc tế/NBSP, giờ có giây, dữ liệu sai, email/URL, tên ngoại ngữ, m², 1e-3']],[90,95,WIDTH-185])
add('Gold được soạn độc lập bằng tay trước các lượt chạy; không lấy output model làm đáp án. Ngày viết bằng dấu / trong tiếng Việt theo ngày/tháng; tiếng Anh dùng quy ước US cho ví dụ 1/4. Giữ nguyên văn bản là chính sách riêng cho 4 ca giờ/ngày không hợp lệ, không phải yêu cầu phổ quát của mọi TTS.')
add('Tiêu chí đạt','h2')
add('So khớp toàn câu với một trong các cách đọc chấp nhận sau khi bỏ khác biệt chữ hoa, dấu câu, khoảng trắng và các biến thể đã khai báo (ngàn/nghìn, linh/lẻ, mươi tư/mươi bốn, “and” tiếng Anh). Chữ số chưa đổi vẫn là chữ số, không tự coi là đúng. Chấm trước acoustic model; không suy ra WER, thanh điệu, độ tự nhiên hoặc MOS.')
add('Bốn ca “foreign” kiểm hợp đồng phiên âm/giữ chữ; các cách phiên âm Việt chỉ là đáp án quy ước, không phải bằng chứng phát âm duy nhất đúng. Mã tiếng Anh “VN” và “v n” được xem là hai chuỗi khác ở phép chấm văn bản, dù TTS có thể tự xử lý một số acronym.','small')
add('Đo latency','h2')
add('Mỗi ca: 1 lượt làm nóng, sau đó 5 lượt đo; thứ tự B0/M1/M2/M3 được xáo bằng seed cố định. p50/p95 gộp 480 quan sát/phương pháp, gồm bảo vệ định danh, bước tên VI và chuẩn hóa; không gồm nạp model, TTS, ASR hay MT. Không cache output ứng viên giữa các lượt. Hai đoạn tổng hợp đo riêng 3 lượt chuẩn hóa và 1 lượt tổng hợp âm thanh/cấu hình.')
add('Đối chiếu trước/sau','h2')
add('Giữ kết quả trước sửa trong before_identifier_policy/results.json. Lượt phụ bỏ bước phiên âm tên nhưng giữ lớp định danh chỉ dùng để tìm ảnh hưởng thứ tự xử lý; số latency của lượt phụ không dùng để xếp hạng. Các tập tự soạn và nhỏ không đại diện phân bố hội thoại thực tế.')
page()
# 6 - results by language and type.
heading(5,'Thống kê chất lượng văn bản')
tbl([['Tập','B0','M1','M2','M3']]+[[l.upper()+' (48 ca)']+[cell(S[m]['by_lang'][l]) for m in ['B0','M1','M2','M3']] for l in ['vi','en']]+[['Tổng 96 ca']+[metric(S[m]) for m in ['B0','M1','M2','M3']]],[105]+[(WIDTH-105)/4]*4)
name={'cardinal':'Số nguyên / số lượng','code':'Mã định danh','date':'Ngày / năm','decimal':'Thập phân / khoa học','electronic':'Email / URL','foreign':'Tên / từ ngoại ngữ','fraction':'Phân số','measure':'Đơn vị đo','money':'Tiền','ordinal':'Số thứ tự','percent':'Phần trăm','plain':'Câu thường','range':'Khoảng số','score':'Tỷ số','telephone':'Điện thoại','time':'Giờ','version':'Phiên bản'}
tbl([['Nhóm (số ca)','B0','M1','M2','M3']]+[[name[k]+f" ({S['M1']['by_kind'][k]['total']})"]+[cell(S[m]['by_kind'][k]) for m in ['B0','M1','M2','M3']] for k in S['M1']['by_kind']],[200]+[(WIDTH-200)/4]*4)
add('B0 cũng đã có lớp bảo vệ định danh. 16/16 điện thoại/mã là kết quả trên tập này; không khẳng định đúng mọi số điện thoại, ZIP quốc tế hoặc mã mới. Nhóm phân số/ngày/phiên bản cho thấy phần còn lại vẫn cần cải thiện ngữ cảnh và grammar.','small')
page()
# 7 - before after / stages.
heading(6,'Thay đổi sau chính sách định danh')
tbl([['Phương pháp','Trước sửa','Sau sửa','Chênh lệch ca đạt']]+[[m,cell(OLD['summary'][m]),cell(S[m]),f"{S[m]['passed']-OLD['summary'][m]['passed']:+d}"] for m in ['B0','M1','M2','M3']],[100,130,130,WIDTH-360])
add('Các điểm trước/sau dùng cùng 96 ca và cách chấm. Bản sau chuyển điện thoại/mã sang chữ trước tên/normalizer; không thay trọng số CRF, DistilBERT hoặc gold. Khác biệt latency trước/sau không phải benchmark tăng tốc có kiểm soát; điện thoại/mã đã đọc xong nên số span cần WFST/LM xử lý cũng giảm.')
tbl([['Mức độ (24 ca)','M1','M2','M3']]+[[k]+[cell(S[m]['by_level'][k]) for m in ['M1','M2','M3']] for k in S['M1']['by_level']],[200]+[(WIDTH-200)/3]*3)
add('Ảnh hưởng thứ tự phiên âm tên','h2')
tbl([['Phương pháp','Có bước tên VI (chính)','Bỏ bước tên VI (phụ)']]+[[m,cell(S[m]),cell(R['raw_format_summary'][m])] for m in ['M1','M2','M3']],[110,190,WIDTH-300])
add('Định danh nay được bảo vệ trước tên nên VN123 không còn thành “vơn...” trước khi chuẩn hóa. Tuy nhiên tên/từ ngoại ngữ và ký hiệu đơn vị có thể vẫn bị bước phiên âm cũ tác động. Phần này cần G2P/lexicon riêng, không thể kết luận đã sửa chỉ nhờ điểm TN tăng.')
add('Đánh giá CRF','h2')
add('M2 tăng một số ca ngày theo ngữ cảnh nhưng còn sai số lượng ở ca khác; tổng điểm bằng M1. Tập huấn luyện tổng hợp còn hẹp và không chứng minh tổng quát hóa ngoài template. Chưa có cơ sở chọn M2 chỉ vì kiến trúc có học máy.')
page()
# 8 - resource accounting + audio.
heading(7,'Latency, dung lượng và audio thực đo')
gram=(sum(E['far_bytes'].values())+sum(v for k,v in E['artifact_bytes'].items() if k.endswith('.fst')))/1e6
crfmb=E['artifact_bytes']['tagger.crfsuite']/1e6
lmmb=sum(E['mlm_bytes'].values())/1e6
sharedtts=(E['piper_model_bytes']+sum(E['supertonic_model_bytes'].values()))/1e6
tbl([['Cấu hình','p50 / p95 TN (ms)','Payload chuẩn hóa*']]+[[m,f"{S[m]['p50_ms']:.2f} / {S[m]['p95_ms']:.2f}",f"{gram+(crfmb if m=='M2' else lmmb if m=='M3' else 0):.2f} MB"] for m in ['M1','M2','M3']],[95,170,WIDTH-265])
add(f'* Payload = FAR NeMo + các FST phụ dùng trong prototype ({gram:.2f} MB); M2 cộng checkpoint CRF {crfmb:.3f} MB; M3 cộng trọng số/tokenizer/config LM {lmmb:.2f} MB. Không gồm Python, OpenFst, scikit-learn, PyTorch, thư viện phụ hoặc model TTS. TTS dùng chung hai ngôn ngữ: khoảng {sharedtts:.2f} MB ONNX. Đây là byte file thực tế, không phải RAM cần lúc chạy.','small')
add(f"Cùng tiến trình nạp nhiều phương pháp và TTS có RSS đỉnh {E['process_peak_rss_mb']:.0f} MB; không dùng số này để gán RAM riêng cho từng phương pháp. Môi trường {E['platform']}, Python {E['python'].split()[0]}, {E['cpu_count']} logical CPU; LM 4 luồng. Không benchmark NPU.",'small')
add('Hai đoạn tổng hợp - audio CPU','h2')
tbl([['Ngôn ngữ / bản','TN p50 (ms)','Tổng hợp (s)','Audio (s)','Số đoạn']]+[[d['lang'].upper()+' / '+d['method'],f"{d['normalization_p50_ms']:.1f}",f"{d['synthesis_seconds']:.2f}",f"{d['audio_seconds']:.2f}",str(d['chunks'])] for d in R['demos']],[105,110,110,105,WIDTH-430])
add('VI: Piper vais1000; EN: Supertonic 3 giọng M1, 8 bước, speed 1.0, seed 521. Dùng ngưỡng chia hiện tại 26 từ VI / 82 ký tự EN; ghi lại đầu vào graph/front-end. Model gốc CPU không tái tạo chính xác graph gộp, giới hạn cửa sổ hay kernel NPU; thời gian tổng hợp ở đây không được thay cho số đo hệ thống thiết bị. Audio chỉ được kiểm kỹ thuật (mẫu hợp lệ, có tín hiệu), chưa nghe mù/chấm ASR độc lập.','small')
page()
# 9 - concrete errors, no subjective listening claims.
heading(8,'Ví dụ kết quả và lỗi còn lại')
for cid in ['VI-017','VI-025','VI-024','VI-035']:
 c=next(c for c in R['cases'] if c['id']==cid)
 add(cid+' - '+c['text'],'h2')
 for m in ['M1','M2','M3']:
  z=c['current_order'][m];add(m+(' [Đạt]: ' if z['pass'] else ' [Chưa đạt]: ')+z['text'],'small')
add('Nhận xét có thể kiểm chứng từ văn bản','h2')
add('Mã đã đọc theo từng ký tự và giữ số 0. Các lỗi còn lại tập trung ở khoảng số bị hiểu thành số âm/ngày, phiên bản bị hiểu thành ngày, phân số bị chọn thành ngày và xử lý tên/URL. Mô hình ngôn ngữ ưu tiên cách diễn đạt có xác suất cao chưa đồng nghĩa giữ đúng ý định người nói.')
quote('a simple FST-based filter can help mitigate those errors','[S3] Abstract, PDF trang 1; ví dụ lỗi và bộ lọc: PDF trang 4, Bảng 2-5.')
add('Gợi ý tiếp theo: mở rộng grammar theo kiểu range/version/electronic; dùng dữ liệu ngữ cảnh thật cho tagger; đánh giá G2P tên riêng riêng biệt; bổ sung kiểm tra giá trị ngày/giờ và đánh giá audio bằng ASR độc lập + nghe mù. Chưa sửa các lỗi này trong lần chạy để tránh vừa chỉnh theo đáp án vừa báo điểm.')
page()
# 10 - sources and reproducibility.
heading(9,'Nguồn, tái lập và giới hạn')
refs=[
 ('S1','Bakhturina, Zhang & Ginsburg (2022). Shallow Fusion of Weighted Finite-State Transducer and Language Model for Text Normalization. Trích Abstract: PDF tr.1 (kỷ yếu 491); kiến trúc: PDF tr.2-3 (492-493).','https://www.isca-archive.org/interspeech_2022/bakhturina22_interspeech.pdf'),
 ('S2','Dang, Vuong & Phan (2022). Non-Standard Vietnamese Word Detection and Normalization for Text-to-Speech. Trích Abstract: PDF tr.1; taxonomy/kiến trúc: tr.2-3; kết quả và hạn chế: tr.5-6.','https://arxiv.org/pdf/2209.02971'),
 ('S3','Sproat & Jaitly (2017). An RNN Model of Text Normalization. Trích Abstract: PDF tr.1; lỗi và FST filter: PDF tr.4, Bảng 2-5.','https://www.isca-archive.org/interspeech_2017/sproat17_interspeech.pdf'),
 ('S4','DistilBERT multilingual cased - model card; checkpoint '+E['mlm_revision']+'. Tài liệu web, không có số trang.','https://huggingface.co/distilbert/distilbert-base-multilingual-cased'),
 ('S5','NeMo Text Processing - mã nguồn grammar. Commit '+E['nemo_commit']+'. Tài liệu web, không có số trang.','https://github.com/NVIDIA/NeMo-text-processing/tree/'+E['nemo_commit']),
]
for key,desc,url in refs:
 add('['+key+'] '+desc,'small')
 add('<link href="'+url+'" color="#176b75">Mở nguồn gốc</link>','small',True)
add('Nguồn nội bộ (đường dẫn tương đối từ thư mục nhánh)','h2')
for t in ['[L1] contest_VNG/s2s_edge/s2s_edge/{tts.py,textnorm.py,names.py,identifiers.py}; docs/tts_format_issues.md.', '[L2] piper/piper_npu/{README.md,.gitignore}; supertonic/supertonic_npu/{README.md,.gitignore}.', '[L3] contest_VNG/s2s_edge/models.dev_beam.json; cấu hình model trỏ đường dẫn Windows.']:
 add(t,'small')
add('Tái lập','h2')
add('Script: research/tts_format_benchmark/rerun_current.py. Bộ ca: test_cases.json; checkpoint CRF: artifacts/tagger.crfsuite; môi trường: requirements-lock.txt. Cần đặt PYTHONPATH trỏ mã NeMo và HF_HOME trỏ cache model như trong script. File results.json lưu commit, hash nguồn/gold/checkpoint, cấu hình model, từng output, từng quan sát latency và đường dẫn audio.','small')
add('Giới hạn kết luận','h2')
add('Tập 96 ca tự soạn có chủ đích stress-test; chưa có CI thống kê, nhiều seed hay dữ liệu người dùng độc lập. Điểm là mức tuân thủ văn bản, không phải chất lượng phát âm. Chưa kiểm NPU hoặc ASR→MT→TTS đầu-cuối. Có thay đổi cục bộ ở lớp định danh; chưa commit/push. So sánh kiến trúc không phải tái lập kết quả công bố của paper.','small')
page()
# 11-12 dense transcripts.
for lang in ['vi','en']:
 heading('A'+('1' if lang=='vi' else '2'),'Đoạn tổng hợp '+('tiếng Việt' if lang=='vi' else 'tiếng Anh'))
 ds=[d for d in R['demos'] if d['lang']==lang]
 add('Đầu vào','h2');add(ds[0]['input'],'transcript')
 add('Bản dưới đây là chữ thực đưa vào TTS sau chính sách định danh và bước tên. Audio nằm trong thư mục kết quả audio/'+lang+'_M1.wav, _M2.wav, _M3.wav; bản _B0.wav dùng để đối chiếu.','small')
 for d in ds:
  if d['method']=='B0':continue
  add(label(d['method']),'h2');add(d['text_for_tts'],'transcript')
 if lang=='en':
  same=next(d for d in ds if d['method']=='M1')['sha256']==next(d for d in ds if d['method']=='M2')['sha256']
  add('M1 và M2 có audio cùng SHA-256 trong đoạn này.' if same else 'M1 và M2 có khác biệt audio trong lần chạy này.','small')
 page()
# 13-20 matrix of all 96 expected outcomes, readable 12 cases/page.
for lang in ['vi','en']:
 for level in sorted(S['M1']['by_level']):
  group=[c for c in R['cases'] if c['lang']==lang and c['level']==level]
  heading('B',f"Toàn bộ ca {lang.upper()} - {level}")
  add('D = đạt kỳ vọng văn bản; X = chưa đạt. Cột Đầu vào / Kỳ vọng ghi một cách đọc chấp nhận; toàn bộ lựa chọn và output từng phương pháp có trong cases.csv và results.json.','small')
  data=[['ID','Đầu vào / Kỳ vọng','B0','M1','M2','M3']]
  for c in group:
   data.append([c['id'],c['text']+'\n→ '+c['expected'][0]]+['D' if c['current_order'][m]['pass'] else 'X' for m in ['B0','M1','M2','M3']])
  # Convert newline to explicit Paragraph breaks in the only textual column.
  table=tbl(data,[44,WIDTH-44-4*29]+[29]*4,small=True)
  add('Nguồn: rerun_current.py, '+E['finished_local'][:19].replace('T',' ')+', CPU. Không dùng bảng này như kết quả nghe audio.','small')
  if not (lang=='en' and level==sorted(S['M1']['by_level'])[-1]):page()

def footer(c,doc):
 c.setStrokeColor(LINE);c.setLineWidth(.5);c.line(MARGIN,36,W-MARGIN,36)
 c.setFont('Arial',8);c.setFillColor(MUTED)
 c.drawString(MARGIN,23,'ONEVOICE · TTS Anh - Việt · 11/11/2026')
 c.drawRightString(W-MARGIN,23,str(doc.page))
 if doc.page>1:
  c.setFont('Arial',7.5);c.drawString(MARGIN,H-26,'KIỂM THỬ THỰC TẾ / CPU / 3 PHƯƠNG PHÁP')

# Real source excerpt provenance, kept short; counts across S1 quotes = 23 words.
q=[dict(id='S1-M1',source='S1',page_pdf=1,quote='Text normalization (TN) systems in production are largely rule-based using weighted finite-state transducers (WFST).'),dict(id='S1-M3',source='S1',page_pdf=1,quote='a neural language model picks the best one'),dict(id='S2-M2',source='S2',page_pdf=1,quote='First, a model-based tagger is designed to detect NSWs.'),dict(id='S3-filter',source='S3',page_pdf=1,quote='a simple FST-based filter can help mitigate those errors')]
(DATA/'source_excerpts.json').write_text(json.dumps(q,ensure_ascii=False,indent=2))
doc=SimpleDocTemplate(str(OUT),pagesize=A4,rightMargin=MARGIN,leftMargin=MARGIN,topMargin=48,bottomMargin=48,title='OneVoice - Báo cáo TTS Anh Việt: ba phương pháp',author='OneVoice / benchmark workspace')
doc.build(flow,onFirstPage=footer,onLaterPages=footer)
r=PdfReader(OUT)
print('PDF',str(OUT),'PAGES',len(r.pages),'BYTES',OUT.stat().st_size)
for i,pag in enumerate(r.pages):
 text=pag.extract_text();print(i+1,len(text),text[:75].replace('\n',' | '))
assert len(r.pages)>=12
