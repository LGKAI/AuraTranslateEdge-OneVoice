from pathlib import Path
import json,csv,statistics,collections,math
r=Path(__file__).resolve().parents[1];out=r/'results/en_vi_audit';data=json.loads((out/'audit.json').read_text())
# Content review of existing transcripts, with independent ASR cross-check.
text='''EV-0157|MT,TTS?|vừa|Bản dịch giữ mốc 1990–2000. heard cũ làm mất mươi, nhưng Whisper nghe lại ra 1990 đúng: chưa kết luận TTS làm sai năm chỉ từ heard.
EV-0182|ASR,MT,REF?|cao|So với text/ref, ý điện thoại không tốt bị đổi thành thật tuyệt vời. Hai ASR đều không nhận rõ câu nguồn; cần người nghe chốt từ phủ định, không quy toàn bộ cho MT.
EV-0228|MT|thấp|Giữ đúng câu hỏi nhắc lịch bác sĩ; báo thức tự nhiên hơn báo động. Không có lỗi nội dung lớn trong bản dịch.
EV-0240|MT,TTS?|vừa|Giữ ý đặt bánh và hỏi tình trạng. Vị trí xin vui lòng chưa tự nhiên; hai ASR nghe lại có sai khác quanh cụm này, cần nghe xác nhận phát âm.
EV-0250|MT,tên riêng|vừa|Phần hỗ trợ/thanh toán còn đúng ý; tên đối tượng nhận thanh toán bị dịch thành khí hóa thạch. Không chấm phát âm tên riêng theo phạm vi đã thống nhất.
EV-0333|MT,TTS?|cao|feeling a bit off là cảm thấy hơi bất thường, bị dịch thành cảm thấy bị cắn. ears buzzing cũng dịch cứng thành tai rung hơi; không thể sửa bằng normalize số.
EV-0358|TN/TTS|cao|ASR giữ đúng 734 và 67297. ZIP cần đọc từng chữ số nhưng heard đọc thành số lượng lớn; Whisper nghe output cũng không giữ chuỗi mã rõ. Tên địa chỉ không đưa vào chấm phát âm. asr_err 28,6 không phản ánh hết việc ASR đã giữ đúng giá trị số.
EV-0383|MT|thấp|Giữ ý đặt lịch khám trực tiếp và không phải cấp cứu; diễn đạt dài, nhiều từ thừa nhưng thông tin chính còn nguyên.
EV-0491|MT|vừa|$10 được giữ. all chip in bị rút thành chúng tôi trả, thiếu sắc thái mỗi người góp một ít tiền xăng.
EV-0523|ASR,MT|cao|Mất lời chia sẻ sorry to hear that, nhận thành Sophia rồi MT dịch theo. Whisper input nhận được sorry to hear that. Tên bị sinh ra không phải tên thật trong text tham chiếu.
EV-0752|MT|cao|get out of this consultation là nhận được/lợi ích từ buổi tư vấn, không phải thoát khỏi cuộc tư vấn.
EV-0894|MT|vừa|moving on swiftly là chuyển sang câu hỏi/phần tiếp theo, bị dịch thành đi nhanh. Câu hỏi về dị ứng vẫn giữ được ý.
EV-1173|ASR,MT|cao|Theo ref, câu nói xe dọn tuyết vùi xe vào thứ Ba; ASR thành snow post bearing và mất Tuesday, MT thành cây tuyết mang xe. Whisper cũng nghe không rõ đầu vào; cần kiểm lại audio gốc trước khi chốt nguyên nhân nhận dạng.
EV-1212|ASR,MT|cao|Câu hỏi về cơn đau liên tục hay lúc có lúc không bị dịch cứng thành làm đau và nó đến và đi. Sai cấu trúc/ngữ cảnh, không phải lỗi format số.
EV-1229|ASR,MT,tên riêng|cao|Đối tượng nhận khoản thanh toán thành falseso guests rồi những khách giả. Phần câu hỗ trợ còn đúng; lỗi tên/đối tượng cần tách khỏi chấm phát âm TTS.
EV-1508|MT|vừa|Khung giờ 12 đến 3 được giữ và heard có từ giờ. Phần tôi sẽ nhìn vào nó như... hoặc cái gì đó là dịch từng chữ, nên diễn đạt lại.
EV-1725|TTS?|vừa|Bản dịch đúng ý chủ nhà hàng/quán sushi. heard ghi sushi thành xùi: từ vay mượn cần nghe xác nhận; chưa coi ASR nghe lại là kết luận chắc chắn.
EV-1829|không có lỗi lớn ở mức text|thấp|Giữ đúng ý đã được giúp đủ và cảm ơn; một trong những bản dịch sát nghĩa nhất.
EV-1863|ASR,MT|cao|get colder bị nhận thành corridor rồi thành đi vào hành lang. Whisper độc lập nghe được get colder trong input.
EV-2177|ASR,MT,REF?|cần nghe|Câu nguồn/text có nhiều chỗ mơ hồ: shame/fair day/Mongey. Bản dịch để làm xấu hổ một ngày đẹp không tự nhiên. Không chốt nội dung tên con vật chỉ từ hai ASR.
EV-2248|không có lỗi lớn ở mức text|thấp|Giữ đúng câu hỏi 10 bước dễ làm giàu; số 10 được heard đọc đúng mười.
EV-2316|MT|vừa|Ý một số người cần thuốc và đây không phải lựa chọn đề xuất ban đầu còn được giữ. first instance thành trường hợp đầu tiên chưa tự nhiên theo ngữ cảnh.
EV-2364|ASR nhẹ,MT|thấp|ASR lặp remotemote nhưng MT vẫn ra điều khiển từ xa. Bản dịch dài/cứng, chưa thấy mất nội dung chính.
EV-2388|MT,TTS?|vừa|Giữ ý mỗi lần đi xe buýt có chuyện xấu; ranh giới câu phủ định chưa tự nhiên. MT để xe bus ở cuối, hai ASR nghe gần xe bạc/bậc; cần nghe xác nhận.
EV-2390|ASR/REF?,MT|cao|Theo ref, walk the dog là dắt chó đi dạo; ASR và Whisper đều nghe work the dog, MT thành việc làm cho con chó. Cần kiểm audio để tách lỗi phát âm/annotation và lỗi ASR.
EV-2401|ASR/REF?,MT|cao|Theo ref, yêu cầu gửi một thẻ mới; ASR thành send me more và MT thành gửi thêm nữa. Whisper nghe send me one, nên mất ý mới cần đối chiếu tai người. Không chấm tên riêng.
EV-2476|không có lỗi lớn ở mức text|thấp|Giữ ý công việc căng thẳng và nhiều việc đang diễn ra. asr_err 21,1 chủ yếu chưa thể dùng làm bằng chứng sai nghĩa vì có nhiều từ đệm trong ref.
EV-2748|INPUT,ASR rỗng|không có đầu ra|Thiếu _out.wav, ASR/MT/heard rỗng. Input RMS khoảng -74 dBFS, 77,48% mẫu bằng 0; Whisper rỗng cả trước và sau tăng peak lên 0,5. Cần kiểm file/thu âm/tiền xử lý trước khi quy lỗi cho model ASR.
EV-2802|không có lỗi lớn ở mức text|thấp|Giữ ý biết ơn công ty/cơ hội dù chưa hoàn hảo. Sai khác biết ơn/biết ở giữa các ASR nghe lại chưa đủ chốt TTS sai.
EV-3057|MT|cao|throw it together trong ngữ cảnh các phần tài liệu là ghép/tổng hợp lại, không phải thả nó lại. Whisper xác nhận cụm nguồn; lỗi nghĩa chủ yếu ở MT.
EV-3248|MT,TTS?|cao|how much time to boil an egg hỏi thời gian luộc trứng, bị đổi thành còn bao lâu để nấu trứng. heard và Whisper đều có dạng phát âm không rõ; tts_err 42,9 cần nghe kiểm, không tự coi là đo chất lượng cảm nhận.
EV-3459|ASR,MT|cao|32 tuổi thành 2 tuổi. Whisper input nhận được 32; lỗi giá trị số bắt đầu ở ASR, normalize TTS không thể tự biết để khôi phục 32.
EV-3775|TN/TTS|cao|MT giữ đúng 5:30 sáng nhưng heard là năm ba sáng, Whisper gần năm ba mươi sáng. Thiếu cách đọc rõ năm giờ ba mươi sáng; không quy mốc giờ này thành lỗi MT chỉ vì ASR ghi at530.
EV-3791|MT|cao|minutes of meeting phải là biên bản cuộc họp, không phải những phút gặp gỡ. ASR giữ cụm nguồn đúng; không phải vấn đề đọc đơn vị phút.
EV-4009|MT|vừa|Giữ ý may mắn có bàn đôi cuối và không đặt trước, nhưng không có chỗ đặt diễn đạt sai/cứng với reservation. Cần đổi thành chưa đặt bàn trước.
EV-4177|ASR nhẹ,MT|thấp|Giữ ý chuyển tiền giữa các tài khoản. Singular account trong ASR làm câu kém chuẩn nhưng tiếng Việt vẫn có thể hiểu nhiều tài khoản từ giữa. Không chấm phát âm tên riêng.
EV-4250|MT nhẹ,TTS?|vừa|Giữ câu hỏi số lần đi vệ sinh/ngày khi tiêu chảy. Bạn sẽ đi hơi lệch sắc thái hiện tại. heard ghi tiêu chả nhưng Whisper nghe tiêu chảy đúng: chưa kết luận TTS sai từ một bộ chấm.
EV-4294|MT|vừa|Twitter handle là tên người dùng/tài khoản, không phải hồ sơ quản lý. Ý hỏi hồ sơ và số người theo dõi vẫn còn. Tên riêng không đưa vào chấm phát âm.
EV-4321|ASR,MT|cao|going to read bị nhận thành glad to read rồi dịch tôi rất vui; Whisper nghe going to. text thành tin nhắn cũng lệch ngữ cảnh. Đại từ you/we cuối câu chưa chốt vì ref và ASR chưa nhất quán.
EV-4505|REF?|cần nghe|Whisper và ASR gốc đều nghe every month you see a new launch, trong khi text/ref viết khác. Không kết luận ASR bịa mỗi tháng chỉ vì WER 24,1; cần sửa/kiểm annotation nguồn.
EV-4810|MT|vừa|local branch hours là giờ làm việc của chi nhánh; MT giữ chủ đề nhưng diễn đạt giờ của chi nhánh là bao giờ thiếu tự nhiên. Không chấm tên riêng.
EV-5187|MT nhẹ|thấp|Giữ ý đang thiếu tiền và không cần trả toàn bộ hóa đơn. No way thành Không thể hơi cứng nhưng nội dung chính đúng.
EV-5269|ASR,MT,INPUT?|cao|Theo ref, không dị ứng thuốc nhưng dị ứng đậu phộng. ASR/MT làm mất thông tin dị ứng, thành không có thuốc nào trong vpin. Whisper cũng không nhận rõ input này; cần nghe/kiểm chất lượng nguồn.
EV-5301|TN/TTS,MT|cao|800 trong ngữ cảnh báo thức đã thành 08:00 đúng ở MT, nhưng heard là tám không; cần tám giờ. alarm nên là báo thức thay vì báo động. tts_err=0 không chứng minh chuẩn hóa giờ đúng.
EV-5455|MT nhẹ|thấp|Giữ câu hỏi triệu chứng hiện còn hay đã hết. Câu lặp ngay bây giờ và dùng hoặc hơi cứng, chưa mất ý chính.
EV-5482|không có lỗi lớn ở mức text|thấp|Câu hỏi có nhớ lúc đó đang làm gì được dịch đúng và tự nhiên.
EV-5600|MT|vừa|Giữ ý dành thời gian với bạn bè và để xem sao; thời gian chất lượng... bên ngoài như là... là dịch từng chữ, nên diễn đạt tự nhiên hơn.
EV-5621|MT,TTS?|cao|twelve noon/12 PM là 12 giờ trưa, MT đổi thành 12 giờ tối: sai thời điểm. Đây là lỗi nghĩa ở MT, không thể chỉ đổi cách đọc số ở TTS. Whisper output không rõ thêm phần giờ, cần nghe xác nhận phát âm.
EV-5655|không có lỗi lớn ở mức text|thấp|Bản dịch giữ đúng ý đăng trạng thái tận hưởng thời tiết đẹp. Tên riêng không chấm trong phạm vi này.
EV-5674|MT nhẹ,TTS?|thấp|MT giữ ý xin bản tóm tắt tình hình. heard/Whisper khác nhau ở từ/tên cuối, không chấm tên riêng và chưa kết luận lỗi âm thanh tuyệt đối.'''
notes={}
for line in text.splitlines():
 id,stage,severity,note=line.split('|',3);notes[id]={'stages':stage,'priority':severity,'assessment':note}
assert len(notes)==50 and set(notes)=={c['id'] for c in data['cases']}
for c in data['cases']:c['review']=notes[c['id']]
rows=list(csv.DictReader(open(r.parent/'en-vi/BTC_results.csv',encoding='utf-8-sig')))
metrics={}
for k in ['asr_err','chrf','tts_err','total_s','mt_s','tts_s']:
 vals=[float(z[k]) for z in rows if z[k]];metrics[k]={'n':len(vals),'mean':statistics.mean(vals),'median':statistics.median(vals)}
summary={'cases':50,'input_wav':50,'output_wav':49,'missing_output':['EV-2748'],'adv_cases':0,'all_existing_wav_finite':all(v['finite'] for c in data['cases'] for v in c['files'].values() if v.get('exists')),'logged_tts_error_zero':sum(z['tts_err']=='0.0' for z in rows),'logged_asr_error_over_20':sum(float(z['asr_err'])>20 for z in rows),'logged_flags':{k:sum(k in z['issues'].split() for z in rows) for k in ['asr_bad','tts_bad','slow']},'logged_metrics_not_human_quality':metrics}
data['summary']=summary;data['names_policy']='Không chấm phát âm tên riêng theo yêu cầu trước; có ghi nhận việc MT thay đổi đối tượng/nội dung khi cần.'
data['limitations']='Kiểm tra nội dung dựa trên text/ref, log và ASR độc lập, không phải chấm nghe mù của người. Có trường hợp reference và hai ASR không thống nhất; đã đánh dấu cần nghe. Không có phiên bản model/normalizer đủ để xác nhận đây là bản sửa mới. Không chạy lại S2S và không sửa các file gốc.'
(out/'review.json').write_text(json.dumps(data,ensure_ascii=False,indent=2));(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
lines=['# Kiểm tra en-vi: 50 ca Anh → Việt','', 'Nguồn: `en-vi/BTC_results.csv` và `en-vi/BTC/`. ADV rỗng. Đã đọc 50 JSON, kiểm tra 99 WAV hiện có và chạy ASR Whisper Small độc lập trên audio (kết quả cũ chỉ tái sử dụng nếu hash audio trùng). File gốc không thay đổi.','', '## Kết luận','', '- Có 49 audio output; EV-2748 không có đầu ra. Input ca này rất nhỏ, khoảng -74 dBFS và 77,48% mẫu bằng 0; ASR độc lập vẫn rỗng sau tăng âm lượng.','- Chưa đạt toàn bộ: có lỗi ASR làm mất số/ý, lỗi dịch thành ngữ và lỗi đọc giờ/ZIP.','- 24/49 ca ghi tts_err=0 nhưng đây không phải chứng nhận phát âm hoặc nghĩa đúng. EV-5301 là ví dụ đọc giờ chưa đúng dù chỉ số bằng 0.','- Tên riêng không chấm phát âm. ASR nghe lại chỉ là bằng chứng đối chiếu; không gọi đây là đánh giá nghe mù hay MOS.','- Các file không ghi đủ phiên bản model/normalizer, chưa xác nhận đã chạy qua bản sửa mới.','', '## Thống kê từ CSV (không phải điểm nghe của người)','',f'- ASR error trung bình: {metrics["asr_err"]["mean"]:.2f}% trên 50 ca; số này có thể chịu ảnh hưởng từ đệm và cách ghi số.',f'- chrF bản dịch trung bình: {metrics["chrf"]["mean"]:.2f}/100 trên 49 ca có bản dịch; không dùng để tự quyết định một câu đúng nghĩa.',f'- TTS error trong log trung bình: {metrics["tts_err"]["mean"]:.2f}% trên 49 ca. Không tương đương MOS.',f'- Thời gian log trung bình: {metrics["total_s"]["mean"]:.2f} giây/ca (gồm ca rỗng); 43 ca có cờ slow. Không rõ đủ thiết bị/cấu hình để so trực tiếp với lần chạy khác.','', '## Nhận xét từng ca','']
for c in data['cases']:
 d=c['metadata'];n=c['review'];lines.extend([f'### {c["id"]} — {n["priority"]} / {n["stages"]}', '',n['assessment'],'',f'- Input tham chiếu: {d["text"]}',f'- ASR trong log: {d["asr"] or "(rỗng)"}',f'- Bản dịch: {d["mt"] or "(rỗng)"}',f'- Whisper nghe input: {c.get("independent_in",{}).get("text") or "(rỗng)"}',f'- Whisper nghe output: {c.get("independent_out",{}).get("text") or "(không có output)"}',''])
(out/'review.md').write_text('\n'.join(lines));print(json.dumps(summary,ensure_ascii=False,indent=2))
