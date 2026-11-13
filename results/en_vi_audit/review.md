# Kiểm tra en-vi: 50 ca Anh → Việt

Nguồn: `en-vi/BTC_results.csv` và `en-vi/BTC/`. ADV rỗng. Đã đọc 50 JSON, kiểm tra 99 WAV hiện có và chạy ASR Whisper Small độc lập trên audio (kết quả cũ chỉ tái sử dụng nếu hash audio trùng). File gốc không thay đổi.

## Kết luận

- Có 49 audio output; EV-2748 không có đầu ra. Input ca này rất nhỏ, khoảng -74 dBFS và 77,48% mẫu bằng 0; ASR độc lập vẫn rỗng sau tăng âm lượng.
- Chưa đạt toàn bộ: có lỗi ASR làm mất số/ý, lỗi dịch thành ngữ và lỗi đọc giờ/ZIP.
- 24/49 ca ghi tts_err=0 nhưng đây không phải chứng nhận phát âm hoặc nghĩa đúng. EV-5301 là ví dụ đọc giờ chưa đúng dù chỉ số bằng 0.
- Tên riêng không chấm phát âm. ASR nghe lại chỉ là bằng chứng đối chiếu; không gọi đây là đánh giá nghe mù hay MOS.
- Các file không ghi đủ phiên bản model/normalizer, chưa xác nhận đã chạy qua bản sửa mới.

## Thống kê từ CSV (không phải điểm nghe của người)

- ASR error trung bình: 15.33% trên 50 ca; số này có thể chịu ảnh hưởng từ đệm và cách ghi số.
- chrF bản dịch trung bình: 55.46/100 trên 49 ca có bản dịch; không dùng để tự quyết định một câu đúng nghĩa.
- TTS error trong log trung bình: 6.23% trên 49 ca. Không tương đương MOS.
- Thời gian log trung bình: 6.95 giây/ca (gồm ca rỗng); 43 ca có cờ slow. Không rõ đủ thiết bị/cấu hình để so trực tiếp với lần chạy khác.

## Nhận xét từng ca

### EV-0157 — vừa / MT,TTS?

Bản dịch giữ mốc 1990–2000. heard cũ làm mất mươi, nhưng Whisper nghe lại ra 1990 đúng: chưa kết luận TTS làm sai năm chỉ từ heard.

- Input tham chiếu: play only all music released between the year one thousand nine hundred and ninety and two thousand
- ASR trong log: Play only all music released between the year 1990 and 2000.
- Bản dịch: Chỉ chơi tất cả nhạc phát hành từ năm 1990 đến năm 2000.
- Whisper nghe input: Play only all music released between the year 1900 and 90 and 2000.
- Whisper nghe output: chỉ chơi tất cả nhạc phát hành từ năm 1990 đến năm 2000.

### EV-0182 — cao / ASR,MT,REF?

So với text/ref, ý điện thoại không tốt bị đổi thành thật tuyệt vời. Hai ASR đều không nhận rõ câu nguồn; cần người nghe chốt từ phủ định, không quy toàn bộ cho MT.

- Input tham chiếu: well at one point yes but the storage on my phone not so wonderful
- ASR trong log: Well, at one point, yes, what? The storage I'm for, that's so wonderful.
- Bản dịch: Vâng, ở một thời điểm nào đó, đúng là gì? Khu vực lưu trữ mà tôi muốn, thật tuyệt vời.
- Whisper nghe input: Well, at one point, yes, but the story drama for that's so wonderful.
- Whisper nghe output: Vâng ở một thời điểm nào đó đúng là gì? Cô vực lưu trữ mà tôi muốn thật tuyệt vời.

### EV-0228 — thấp / MT

Giữ đúng câu hỏi nhắc lịch bác sĩ; báo thức tự nhiên hơn báo động. Không có lỗi nội dung lớn trong bản dịch.

- Input tham chiếu: did i remember to set a reminder alarm for my next doctor's appointment
- ASR trong log: Did I remember to set a reminder alarm for my next doctor's appointment.
- Bản dịch: Tôi có nhớ để đặt một báo động nhắc nhở cho cuộc hẹn bác sĩ tiếp theo của tôi không?
- Whisper nghe input: Did I remember to set a reminder alarm for my next doctor's appointment?
- Whisper nghe output: Tôi có nhớ để đặt một báo động nhật nhờ cho cuộc hẹn bác sĩ tiếp theo của tôi không?"

### EV-0240 — vừa / MT,TTS?

Giữ ý đặt bánh và hỏi tình trạng. Vị trí xin vui lòng chưa tự nhiên; hai ASR nghe lại có sai khác quanh cụm này, cần nghe xác nhận phát âm.

- Input tham chiếu: remember i have ordered a cake for tonight's birthday party let me know the status please
- ASR trong log: Remember, I've ordered a cake for tonight's birthday party. Let me know the status, please.
- Bản dịch: Hãy nhớ, tôi đã đặt một chiếc bánh cho bữa tiệc sinh nhật tối nay. Cho tôi biết tình trạng, xin vui lòng.
- Whisper nghe input: Remember I've ordered a cake for tonight's birthday party. Let me know the status please.
- Whisper nghe output: Hãy nhớ, tôi đã đặt một chiếc bánh cho bữa tiệc sinh nhật tối nay cho tôi biết tình trạng, sinh vù lòng".

### EV-0250 — vừa / MT,tên riêng

Phần hỗ trợ/thanh toán còn đúng ý; tên đối tượng nhận thanh toán bị dịch thành khí hóa thạch. Không chấm phát âm tên riêng theo phạm vi đã thống nhất.

- Input tham chiếu: okey doke so we will send your payment to fossil gas is there anything else i can help you with today
- ASR trong log: Okey do, so we will send your payment to fossil gas, is there anything else I can help you with today?
- Bản dịch: Được rồi, chúng tôi sẽ gửi khoản thanh toán của bạn cho khí hóa thạch, có gì khác tôi có thể giúp anh hôm nay không?
- Whisper nghe input: Oh, okay, dope. So we will send your payment to fossil gas. Is there anything else I can help you with today?
- Whisper nghe output: được rồi chúng tôi sẽ gửi khoản thanh toán của bạn cho khí hóa thạch. Có gì khác tôi có thể giúp anh hôm nay không?"

### EV-0333 — cao / MT,TTS?

feeling a bit off là cảm thấy hơi bất thường, bị dịch thành cảm thấy bị cắn. ears buzzing cũng dịch cứng thành tai rung hơi; không thể sửa bằng normalize số.

- Input tham chiếu: and my ears are buzzing a bit i am just they are feeling a bit off
- ASR trong log: And my ears are buzzing a bit, I just feeling bit off.
- Bản dịch: Và đôi tai của tôi rung hơi, tôi chỉ cảm thấy bị cắn.
- Whisper nghe input: and my ears are buzzing a bit, just feeling a bit off.
- Whisper nghe output: Và đôi tai của tôi dung hơi, tôi chỉ cảm thấy bị cánh."

### EV-0358 — cao / TN/TTS

ASR giữ đúng 734 và 67297. ZIP cần đọc từng chữ số nhưng heard đọc thành số lượng lớn; Whisper nghe output cũng không giữ chuỗi mã rõ. Tên địa chỉ không đưa vào chấm phát âm. asr_err 28,6 không phản ánh hết việc ASR đã giữ đúng giá trị số.

- Input tham chiếu: seven hundred thirty four main street harper valley oregon six seven two nine seven
- ASR trong log: 734 Main Street, Harper Valley Oregon 67297.
- Bản dịch: 734 Main Street, Harper Valley Oregon 67297.
- Whisper nghe input: 734 Main Street, Harper Valley, Oregon, 67297.
- Whisper nghe output: 734 ngày thích học phải yên 617.290 bài.

### EV-0383 — thấp / MT

Giữ ý đặt lịch khám trực tiếp và không phải cấp cứu; diễn đạt dài, nhiều từ thừa nhưng thông tin chính còn nguyên.

- Input tham chiếu: so um what i think we could do is um maybe if we book you in for a face to face appointment in one of our clinics it's not an emergency
- ASR trong log: So what I think we could do is. Maybe if we book you in for a face to face appointment in one of our clinics, It's not an emergency.
- Bản dịch: Vì vậy, tôi nghĩ chúng ta có thể làm điều này. Có lẽ nếu chúng tôi đặt bạn vào một cuộc hẹn trực tiếp ở một trong các phòng khám của chúng tôi, đó không phải là một trường hợp khẩn cấp.
- Whisper nghe input: So, what I think we could do is, maybe if we book you in for a face-to-face appointment in one of our clinics, it's not an emergency.
- Whisper nghe output: Vì vậy, tôi nghĩ chúng ta có thể làm điều này, có lẽ nếu chúng tôi đặt bạn vào một cuộc hẹn trực tiếp ở một trong các phòng khám của chúng tôi. Đó không phải là một trường hợp khẩn cấp.

### EV-0491 — vừa / MT

$10 được giữ. all chip in bị rút thành chúng tôi trả, thiếu sắc thái mỗi người góp một ít tiền xăng.

- Input tham chiếu: you'll just need roughly ten dollars for food and gas money my friend is driving so we usually all chip in a few dollars for gas money
- ASR trong log: You'll just need roughly $10 for food and gas money. My friend is driving, so we usually all chip in a few dollars for gas money.
- Bản dịch: Bạn chỉ cần khoảng 10 đô la cho tiền lương thực và xăng. Bạn tôi đang lái xe, vì vậy chúng tôi thường trả vài đô la cho tiền xăng.
- Whisper nghe input: You'll just need roughly $10 for food and gas money. My friend is driving, so we usually all chip in a few dollars for gas money.
- Whisper nghe output: bạn chỉ cần khoảng 10 đô la cho tiền lương thực và xăng. Bạn tôi đang lái xẹ vì vậy chúng tôi thường trả vài đô la cho tiền xăng.

### EV-0523 — cao / ASR,MT

Mất lời chia sẻ sorry to hear that, nhận thành Sophia rồi MT dịch theo. Whisper input nhận được sorry to hear that. Tên bị sinh ra không phải tên thật trong text tham chiếu.

- Input tham chiếu: i am sorry to hear that um how old was he when that happened
- ASR trong log: Sophia, old is he when that happen?
- Bản dịch: Sophia, cậu ta đã bao nhiêu tuổi khi chuyện đó xảy ra?
- Whisper nghe input: Sorry to hear that. How old is he when that happens?
- Whisper nghe output: sau khi cậu ta đã bao nhiêu tuổi khi chuyện đó xảy ra.

### EV-0752 — cao / MT

get out of this consultation là nhận được/lợi ích từ buổi tư vấn, không phải thoát khỏi cuộc tư vấn.

- Input tham chiếu: yeah anything that you had hoped particularly to get out of having this consultation
- ASR trong log: Yeah, anything that you'd hoped, particularly to get out of having this consultation.
- Bản dịch: Ừ, bất cứ điều gì mà anh hy vọng, đặc biệt là thoát khỏi cuộc tư vấn này.
- Whisper nghe input: Yeah, anything that you'd hoped particularly to get out of having this consultation
- Whisper nghe output: bất cứ điều gì mà anh hy vọng, đặc biệt là thoát khỏi cuộc tư vấn này.

### EV-0894 — vừa / MT

moving on swiftly là chuyển sang câu hỏi/phần tiếp theo, bị dịch thành đi nhanh. Câu hỏi về dị ứng vẫn giữ được ý.

- Input tham chiếu: okay all right and just moving on swiftly sir do you have any allergies at all
- ASR trong log: Okay, all right. And just moving on swiftly, Sir, do you have any allergies at all.
- Bản dịch: Được rồi, được rồi. Và chỉ đi nhanh thôi, thưa ngài, ngài có bất kỳ dị ứng nào không?
- Whisper nghe input: Okay, all right. I'm just moving on swiftly sir. Do you have any allergies at all?
- Whisper nghe output: được rồi, được rồi và chỉ đi nhanh thôi. Thương 2, ngày có bất kì dĩ ứng nào không?"

### EV-1173 — cao / ASR,MT

Theo ref, câu nói xe dọn tuyết vùi xe vào thứ Ba; ASR thành snow post bearing và mất Tuesday, MT thành cây tuyết mang xe. Whisper cũng nghe không rõ đầu vào; cần kiểm lại audio gốc trước khi chốt nguyên nhân nhận dạng.

- Input tham chiếu: am i going to have to worry about the snowplows burying my car in the driveway tuesday
- ASR trong log: Am I going to have to worry about a snow post bearing my car in the driveway to.
- Bản dịch: Tôi có phải lo lắng về một cây tuyết mang xe của tôi vào lối vào không?
- Whisper nghe input: Am I going to have to worry about a snow post bearing my car in the driveway to stay?
- Whisper nghe output: tôi có phải lo lắng về một cây tứt mang xe của tôi vào lối vào không."

### EV-1212 — cao / ASR,MT

Câu hỏi về cơn đau liên tục hay lúc có lúc không bị dịch cứng thành làm đau và nó đến và đi. Sai cấu trúc/ngữ cảnh, không phải lỗi format số.

- Input tham chiếu: okay and is the pain is that is it there all the time or does it come and go
- ASR trong log: Okay, and does a pain,, is it there all the time or does it come and go.
- Bản dịch: Được rồi, và làm đau, nó luôn ở đó hay nó đến và đi.
- Whisper nghe input: Okay, and is there pain all the time or does it come and go?
- Whisper nghe output: được rồi và làm đau nó luôn ở đó hay nó đến và đi.

### EV-1229 — cao / ASR,MT,tên riêng

Đối tượng nhận khoản thanh toán thành falseso guests rồi những khách giả. Phần câu hỗ trợ còn đúng; lỗi tên/đối tượng cần tách khỏi chấm phát âm TTS.

- Input tham chiếu: so we will send your payment to fossil gas is there anything else i can help you with
- ASR trong log: So we will send your payment to falseso guests, is there anything else I can help you with?
- Bản dịch: Vậy chúng tôi sẽ gửi tiền của bạn cho những khách giả, có gì khác tôi có thể giúp được không?
- Whisper nghe input: So we will send your payment to fossil gas. Is there anything else that can help you with?
- Whisper nghe output: Vậy chúng tôi sẽ khử tiền của bạn cho những khách già có gì khác tôi có thể giúp được không?

### EV-1508 — vừa / MT

Khung giờ 12 đến 3 được giữ và heard có từ giờ. Phần tôi sẽ nhìn vào nó như... hoặc cái gì đó là dịch từng chữ, nên diễn đạt lại.

- Input tham chiếu: i'll uh look at it like when my usual working hours are like from twelve o clock till three o clock or something
- ASR trong log: I'll look at it like when my usual working hours are like from 12 o'clock to 3 o'clock or something.
- Bản dịch: Tôi sẽ nhìn vào nó như khi giờ làm việc bình thường của tôi là từ 12 giờ đến 3 giờ hoặc cái gì đó.
- Whisper nghe input: I'll look at it like when my usual working hours are like from 12 o'clock to 3 o'clock or something.
- Whisper nghe output: Tôi sẽ nhìn vào nó như khi giờ làm việc bình thường của tôi là từ 10h đến 3h hoặc cái gì đó.

### EV-1725 — vừa / TTS?

Bản dịch đúng ý chủ nhà hàng/quán sushi. heard ghi sushi thành xùi: từ vay mượn cần nghe xác nhận; chưa coi ASR nghe lại là kết luận chắc chắn.

- Input tham chiếu: um i work as a restaurant owner so i own a sushi shop
- ASR trong log: I work as a restaurant owner, so I own a sushi shop.
- Bản dịch: Tôi làm chủ nhà hàng, vì vậy tôi sở hữu một cửa hàng sushi.
- Whisper nghe input: I work as a restaurant owner, so I own a sushi shop.
- Whisper nghe output: tôi làm chủ nhà hàng, vì vậy tôi sở thiệu một cửa hàng suy".

### EV-1829 — thấp / không có lỗi lớn ở mức text

Giữ đúng ý đã được giúp đủ và cảm ơn; một trong những bản dịch sát nghĩa nhất.

- Input tham chiếu: no that's all i needed help with thank you very much for your help
- ASR trong log: No that's all I needed help with, thank you very much for your help.
- Bản dịch: Không, đó là tất cả những gì tôi cần giúp đỡ, cảm ơn rất nhiều vì sự giúp đỡ của bạn.
- Whisper nghe input: No, that's all I needed to help with. Thank you very much for your help.
- Whisper nghe output: Không, nó là tất cả những gì tôi cần giúp đỡ cảm ấn rất nhiều vì sự giúp đỡ của bạn."

### EV-1863 — cao / ASR,MT

get colder bị nhận thành corridor rồi thành đi vào hành lang. Whisper độc lập nghe được get colder trong input.

- Input tham chiếu: will the temperature stay about the same or is it going to get colder
- ASR trong log: Will the temperature stay about the same or is it going to that corridor though?
- Bản dịch: Nhiệt độ sẽ không thay đổi hay nó sẽ đi vào hành lang đó?
- Whisper nghe input: with the temperature stay about the same or is it going to get colder?
- Whisper nghe output: Dịch độ sẽ không thay đổi hay nó sẽ đi vào hành lang đó.

### EV-2177 — cần nghe / ASR,MT,REF?

Câu nguồn/text có nhiều chỗ mơ hồ: shame/fair day/Mongey. Bản dịch để làm xấu hổ một ngày đẹp không tự nhiên. Không chốt nội dung tên con vật chỉ từ hai ASR.

- Input tham chiếu: sure shame fair day and uh how long did you have mongey for
- ASR trong log: To shame fair day and how long did you have monkey for?
- Bản dịch: Để làm xấu hổ một ngày đẹp và anh có con khỉ bao lâu?
- Whisper nghe input: to shame. Fair day. And how long do you have the monkey for?
- Whisper nghe output: để làm số hộ một ngày đẹp và anh có con khỉ bao lâu.

### EV-2248 — thấp / không có lỗi lớn ở mức text

Giữ đúng câu hỏi 10 bước dễ làm giàu; số 10 được heard đọc đúng mười.

- Input tham chiếu: if there were ten easy steps to make myself rich what would they be
- ASR trong log: If there were 10 easy steps to make myself rich, what would they be.
- Bản dịch: Nếu có 10 bước dễ dàng để làm mình giàu, chúng sẽ là gì?
- Whisper nghe input: If there were 10 easy steps to make myself read what would they be?
- Whisper nghe output: Nếu có 10 bước dễ dàng để làm mình sau, chúng sẽ là gì?

### EV-2316 — vừa / MT

Ý một số người cần thuốc và đây không phải lựa chọn đề xuất ban đầu còn được giữ. first instance thành trường hợp đầu tiên chưa tự nhiên theo ngữ cảnh.

- Input tham chiếu: i mean some people with anxiety they do need to take medication to control that but that's not something that we would suggest in the first instance
- ASR trong log: I mean, some people with anxiety, they do need to take medication to control that. But that's not something that we' suggested in the first instance.
- Bản dịch: Ý tôi là, một số người bị lo lắng, họ cần dùng thuốc để kiểm soát điều đó. Nhưng đó không phải là thứ mà chúng tôi đề xuất ở trường hợp đầu tiên.
- Whisper nghe input: I mean some people with anxiety they do need to take medication to control that but that's not something that we've suggested in the first instance.
- Whisper nghe output: ý tôi là một số người bị lo lắng, họ cần dùng thuốc để kiểm soát điều đó, nhưng đó không phải là thứ mà chúng tôi đề xuất ở trường hợp đầu tiên."

### EV-2364 — thấp / ASR nhẹ,MT

ASR lặp remotemote nhưng MT vẫn ra điều khiển từ xa. Bản dịch dài/cứng, chưa thấy mất nội dung chính.

- Input tham chiếu: the biggest frustrations that people found with regard to personal preferences were um something that you you mentioned earlier uh remote controls are often lost in the room
- ASR trong log: The biggest frustrations that people found with regard to personal preferences were something that you, you mentioned earlier. remotemote controls are often lost in the room.
- Bản dịch: Những sự thất vọng lớn nhất mà mọi người tìm thấy về sở thích cá nhân là điều mà bạn, bạn đã đề cập trước đó. Máy điều khiển từ xa thường bị mất trong phòng.
- Whisper nghe input: The biggest frustrations that people found with regard to personal preferences were something that you mentioned earlier. Remote controls are often lost in the room.
- Whisper nghe output: Những sự thất phòng lớn nhất mà mọi người tìm thấy về sở thích cá nhân là điều mà bạn đã đề cập trước đó, máy điều khiển từ xa thường bị mất trong phòng.

### EV-2388 — vừa / MT,TTS?

Giữ ý mỗi lần đi xe buýt có chuyện xấu; ranh giới câu phủ định chưa tự nhiên. MT để xe bus ở cuối, hai ASR nghe gần xe bạc/bậc; cần nghe xác nhận.

- Input tham chiếu: it was not just once every single time i take the bus something bad happens to me or to someone else on the bus
- ASR trong log: It wasn't just once every single time I take the bus, something bad happens to me or to someone else on the bus.
- Bản dịch: Nó không chỉ là một lần mỗi khi tôi lên xe buýt, một cái gì đó xấu xảy ra với tôi hoặc với ai đó khác trên xe bus.
- Whisper nghe input: It wasn't just once. Every single time I take the bus, something bad happens to me or to someone else on the bus.
- Whisper nghe output: Nó không chỉ là một lần mỗi khi tôi lên xe buít, một cái gì đó xấu xảy ra với tôi, hoặc với ai đó khác trên xe bậc.

### EV-2390 — cao / ASR/REF?,MT

Theo ref, walk the dog là dắt chó đi dạo; ASR và Whisper đều nghe work the dog, MT thành việc làm cho con chó. Cần kiểm audio để tách lỗi phát âm/annotation và lỗi ASR.

- Input tham chiếu: i need you to put walk the dog on my list to do
- ASR trong log: I need you to put work the dog on my list to do.
- Bản dịch: Tôi cần anh đưa việc làm cho con chó vào danh sách của tôi.
- Whisper nghe input: I need you to put Work the Dog on my list to do.
- Whisper nghe output: Tôi cần anh đưa Việt làm cho con chó vào danh sách của tôi".

### EV-2401 — cao / ASR/REF?,MT

Theo ref, yêu cầu gửi một thẻ mới; ASR thành send me more và MT thành gửi thêm nữa. Whisper nghe send me one, nên mất ý mới cần đối chiếu tai người. Không chấm tên riêng.

- Input tham chiếu: uhm hi my name's patricia johnson um i lost my debit card can you send me a new one
- ASR trong log: Hi, my name is Patricia Johnson. And I lost my debit card. Can you send me more?
- Bản dịch: Xin chào, tên tôi là Patricia Johnson. Và tôi mất thẻ ghi nợ của mình. Anh có thể gửi thêm nữa không?
- Whisper nghe input: Hi, my name is Patricia Johnson and I lost my debit card. Can you send me one?
- Whisper nghe output: Xin chào tên tôi là Gisorgon, và tôi mất thể khi nợ của mình anh có thể gửi thêm nữa không?

### EV-2476 — thấp / không có lỗi lớn ở mức text

Giữ ý công việc căng thẳng và nhiều việc đang diễn ra. asr_err 21,1 chủ yếu chưa thể dùng làm bằng chứng sai nghĩa vì có nhiều từ đệm trong ref.

- Input tham chiếu: um work's kind of um a bit stressful at the moment i have got a lot going on
- ASR trong log: Work's kind of a bit stressful in the moment, I've got a lot going on.
- Bản dịch: Công việc hơi căng thẳng lúc này, tôi có rất nhiều việc phải làm.
- Whisper nghe input: Work's kind of a bit stressful in the moment. I've got a lot going on.
- Whisper nghe output: Công việc hơi căng thẳng lúc này, tôi có rất nhiều việc phải làm."

### EV-2748 — không có đầu ra / INPUT,ASR rỗng

Thiếu _out.wav, ASR/MT/heard rỗng. Input RMS khoảng -74 dBFS, 77,48% mẫu bằng 0; Whisper rỗng cả trước và sau tăng peak lên 0,5. Cần kiểm file/thu âm/tiền xử lý trước khi quy lỗi cho model ASR.

- Input tham chiếu: your password reset link has has been sent your phone is there anything else i can help you with
- ASR trong log: (rỗng)
- Bản dịch: (rỗng)
- Whisper nghe input: (rỗng)
- Whisper nghe output: (không có output)

### EV-2802 — thấp / không có lỗi lớn ở mức text

Giữ ý biết ơn công ty/cơ hội dù chưa hoàn hảo. Sai khác biết ơn/biết ở giữa các ASR nghe lại chưa đủ chốt TTS sai.

- Input tham chiếu: although i could not say that everything is perfect there i still very much appreciate what the company has given me especially many chances
- ASR trong log: Although I could not say that everything is perfect there, I still very much appreciate what the company has given me, especially many chances.
- Bản dịch: Mặc dù tôi không thể nói rằng mọi thứ ở đó đều hoàn hảo, nhưng tôi vẫn rất biết ơn những gì công ty đã cho tôi, đặc biệt là nhiều cơ hội.
- Whisper nghe input: Although I cannot say that everything is perfect there, I still very much appreciate what the company has given me, especially many chances.
- Whisper nghe output: Mặc dù tôi không thể nói rằng mọi thứ ở đó đều hoàn hảo, nhưng tôi vẫn rất biết ở những gì công ty đã cho tôi, đặc biệt là nhiều cơ hội".

### EV-3057 — cao / MT

throw it together trong ngữ cảnh các phần tài liệu là ghép/tổng hợp lại, không phải thả nó lại. Whisper xác nhận cụm nguồn; lỗi nghĩa chủ yếu ở MT.

- Input tham chiếu: um did someone else want to throw it togeth actually i can probably still throw it together if you want um put your parts into into this one
- ASR trong log: Did someone else wanna throw it together. Actually, I can probably still throw it together if you want, put your parts into, into this one.
- Bản dịch: Có ai khác muốn thả nó lại không? Trên thực tế, tôi có thể đưa nó lại với nhau nếu bạn muốn, đưa phần của bạn vào, vào cái này.
- Whisper nghe input: Did someone else want to throw it together? Actually, I can probably still throw it together if you want. Put your parts into this one.
- Whisper nghe output: Có ai khác muốn thả nó lại không? Trên thực tế, tôi có thể đưa nó lại với nhau nếu bạn muốn đưa phần của bạn vào, vào cái này.

### EV-3248 — cao / MT,TTS?

how much time to boil an egg hỏi thời gian luộc trứng, bị đổi thành còn bao lâu để nấu trứng. heard và Whisper đều có dạng phát âm không rõ; tts_err 42,9 cần nghe kiểm, không tự coi là đo chất lượng cảm nhận.

- Input tham chiếu: how much time do i have to boil an egg
- ASR trong log: How much time do I have to boil an egg.
- Bản dịch: Tôi còn bao lâu để nấu trứng?
- Whisper nghe input: How much time do I have to boil an egg?
- Whisper nghe output: Tôi còn báo lô để nấu trứng."

### EV-3459 — cao / ASR,MT

32 tuổi thành 2 tuổi. Whisper input nhận được 32; lỗi giá trị số bắt đầu ở ASR, normalize TTS không thể tự biết để khôi phục 32.

- Input tham chiếu: thirty two years old okay could you tell me the first line of your address and your postcode please
- ASR trong log: Thattter two years old. Ok, could you tell me the first line of your address and your postcode, please.
- Bản dịch: Đó là 2 tuổi. Được rồi, anh có thể cho tôi biết dòng đầu tiên của địa chỉ và mã bưu điện của anh được không?
- Whisper nghe input: 32 years old. Okay. Could you tell me the first line of your address on your postcode please?
- Whisper nghe output: Đó là hai tuổi được rồi. Anh có thể cho tôi biết rằng đầu tiên của địa chỉ và mã biêu điện của anh được không?

### EV-3775 — cao / TN/TTS

MT giữ đúng 5:30 sáng nhưng heard là năm ba sáng, Whisper gần năm ba mươi sáng. Thiếu cách đọc rõ năm giờ ba mươi sáng; không quy mốc giờ này thành lỗi MT chỉ vì ASR ghi at530.

- Input tham chiếu: get me a taxi to take me to the airport on friday at five thirty in the morning
- ASR trong log: Get me a taxi to take me to the airport on Friday, at530 AM.
- Bản dịch: Đưa cho tôi một chiếc taxi đưa tôi đến sân bay vào thứ Sáu lúc 5:30 sáng.
- Whisper nghe input: Get me a taxi to take me to the airport on Friday at 530am.
- Whisper nghe output: đưa cho tôi một chiếc thắc sĩ đưa tôi đến sân bay vào thứ 6 lúc năm 30 sáng.

### EV-3791 — cao / MT

minutes of meeting phải là biên bản cuộc họp, không phải những phút gặp gỡ. ASR giữ cụm nguồn đúng; không phải vấn đề đọc đơn vị phút.

- Input tham chiếu: so thanks for coming and uh i'll send you minutes of meeting and uh please put your all information in the sharing folder so everybody can share the information
- ASR trong log: So thanks for coming. And I'll send you minutes of meeting. and please put your all information in the sharing folder so everybody can share the information.
- Bản dịch: Cảm ơn đã đến. Và tôi sẽ gửi cho anh những phút gặp gỡ. và xin vui lòng đưa tất cả thông tin của bạn vào thư mục chia sẻ để mọi người có thể chia sẻ thông tin.
- Whisper nghe input: So thanks for coming and I'll send you minutes of meeting and please put your all information in the sharing folder so everybody can share the information.
- Whisper nghe output: Cảm ơn đã đến và tôi sẽ gửi cho anh những phút gặp cỡ và xin vui lòng đưa tất cả thông tin của bạn vào thương mục chi xẻ để mọi người có thể chia sẻ thông tin.

### EV-4009 — vừa / MT

Giữ ý may mắn có bàn đôi cuối và không đặt trước, nhưng không có chỗ đặt diễn đạt sai/cứng với reservation. Cần đổi thành chưa đặt bàn trước.

- Input tham chiếu: we really were lucky we got the last available table for two and we did not even have a reservation did you see the long lines behind us
- ASR trong log: We really were lucky we got the last available table for two and we didn't even have a reservation. did you see the long lines behind us?
- Bản dịch: Chúng tôi thật sự may mắn khi có bàn cuối cùng dành cho hai người và chúng tôi thậm chí không có chỗ đặt. Anh có thấy hàng dài phía sau chúng tôi không?
- Whisper nghe input: We really were lucky. We got the last available table for two, and we didn't even have a reservation. Did you see the long lines behind us?
- Whisper nghe output: Chúng tôi thật sự may mắn khi có bản cuối cùng dành cho hai người và chúng tôi thậm chí không có chỗ đặt. Anh có thấy hàng dài phía sau chúng tôi không?"

### EV-4177 — thấp / ASR nhẹ,MT

Giữ ý chuyển tiền giữa các tài khoản. Singular account trong ASR làm câu kém chuẩn nhưng tiếng Việt vẫn có thể hiểu nhiều tài khoản từ giữa. Không chấm phát âm tên riêng.

- Input tham chiếu: hi my name is michael wilson um i would like to transfer money between my accounts
- ASR trong log: Hi, my name is Michael Wilson, I would like to transfer money between my account.
- Bản dịch: Xin chào, tên tôi là Michael Wilson, tôi muốn chuyển tiền giữa tài khoản của mình.
- Whisper nghe input: Hi, my name is Michael Wilkins. I would like to transfer money between my accounts.
- Whisper nghe output: Xin chào tên tồ là Michael Oizin, tôi muốn chuyển tiền giữa tai khoản của mình.

### EV-4250 — vừa / MT nhẹ,TTS?

Giữ câu hỏi số lần đi vệ sinh/ngày khi tiêu chảy. Bạn sẽ đi hơi lệch sắc thái hiện tại. heard ghi tiêu chả nhưng Whisper nghe tiêu chảy đúng: chưa kết luận TTS sai từ một bộ chấm.

- Input tham chiếu: okay right and with the diarrhea how many times a day are you going to the toilet
- ASR trong log: Okay, right. And with the diarrhea, how many times a day are you going to the toilet.
- Bản dịch: Được rồi, đúng rồi. Và với tiêu chảy, bạn sẽ đi vệ sinh bao nhiêu lần mỗi ngày?
- Whisper nghe input: Okay, right and with the diarrhea how many times a day are you going to the toilet?
- Whisper nghe output: được rồi, búng rồi và với tiêu chảy, bạn sẽ đi về sinh bao nhiêu lần mỗi ngày.

### EV-4294 — vừa / MT

Twitter handle là tên người dùng/tài khoản, không phải hồ sơ quản lý. Ý hỏi hồ sơ và số người theo dõi vẫn còn. Tên riêng không đưa vào chấm phát âm.

- Input tham chiếu: give me details of my twitter handle profile including the total number of followers
- ASR trong log: Give me details of my Twitter handle profile, including the total number of follow.
- Bản dịch: Cho tôi chi tiết về hồ sơ quản lý Twitter của tôi, bao gồm cả tổng số người theo dõi.
- Whisper nghe input: Give me details of my Twitter handle profile including the total number of followers.
- Whisper nghe output: cho tôi chi tiết về hồ sơ quản lý tự ưu tơ của tôi, bao gồm cả tổng số người theo dõi.

### EV-4321 — cao / ASR,MT

going to read bị nhận thành glad to read rồi dịch tôi rất vui; Whisper nghe going to. text thành tin nhắn cũng lệch ngữ cảnh. Đại từ you/we cuối câu chưa chốt vì ref và ASR chưa nhất quán.

- Input tham chiếu: okay um i am going to read out the text now so we will understand what we are supposed to do
- ASR trong log: Okay, I'm glad to read out the text now so we understand what you're supposed to do.
- Bản dịch: Được rồi, tôi rất vui khi đọc tin nhắn này để chúng ta hiểu bạn phải làm gì.
- Whisper nghe input: Okay, I'm going to read out the text now so we understand what you're supposed to do.
- Whisper nghe output: Được rồi, tôi rất vui khi đọc tình nhấn này để chúng ta hiểu bạn phải làm gì."

### EV-4505 — cần nghe / REF?

Whisper và ASR gốc đều nghe every month you see a new launch, trong khi text/ref viết khác. Không kết luận ASR bịa mỗi tháng chỉ vì WER 24,1; cần sửa/kiểm annotation nguồn.

- Input tham chiếu: surprising to me is like uh people give a lot of attention to modifying mobile phones like a real want to see a new launch or something like that
- ASR trong log: Surprising thing is like people give a lot of attention to modifying mobile phones like every month, you see a new launch or something like that.
- Bản dịch: Điều đáng ngạc nhiên là mọi người chú ý rất nhiều đến việc sửa đổi điện thoại di động như mỗi tháng, bạn thấy một sự ra mắt mới hoặc cái gì đó như vậy.
- Whisper nghe input: surprising thing is like people give a lot of attention to modifying mobile phones like every month you see a new launch or something like that
- Whisper nghe output: Điều đắng ngạc nhiên là mọi người chủ ý rất nhiều đến việc sửa đổi điện thoại si động như mỗi tháng. Bạn thấy một sự ra mắt mới hoặc cái gì đó như vậy.

### EV-4810 — vừa / MT

local branch hours là giờ làm việc của chi nhánh; MT giữ chủ đề nhưng diễn đạt giờ của chi nhánh là bao giờ thiếu tự nhiên. Không chấm tên riêng.

- Input tham chiếu: hi my name is elizabeth rodriguez what are the local branch hours
- ASR trong log: Hi, my name is Elizabeth Rodriguez, what are the local branch hours?
- Bản dịch: Xin chào, tên tôi là Elizabeth Rodriguez, giờ của chi nhánh địa phương là bao giờ?
- Whisper nghe input: Hi, my name is Elizabeth Rodriguez. What are the local branch hours?
- Whisper nghe output: Xin chào tênh tổ Lái Lê Dùmê Giá Kỳ dợ, giờ của chi nhánh địa phương là bao giờ?

### EV-5187 — thấp / MT nhẹ

Giữ ý đang thiếu tiền và không cần trả toàn bộ hóa đơn. No way thành Không thể hơi cứng nhưng nội dung chính đúng.

- Input tham chiếu: no way i know that you are a little short on cash at the moment there's no need for you to pay for the whole bill
- ASR trong log: No way. I know that you're a little short on cash at the moment. There's no need for you to pay for the whole bill.
- Bản dịch: Không thể. Tôi biết anh đang thiếu tiền mặt lúc này. Không cần phải trả tiền cho toàn bộ hóa đơn.
- Whisper nghe input: No way, I know that you're a little short on cash at the moment. There's no need for you to pay for the whole bill.
- Whisper nghe output: Không thể tôi biết anh đang thiếu tiền mặt lúc này không cần phải trả tiền cho toàn bộ hóa đơn.

### EV-5269 — cao / ASR,MT,INPUT?

Theo ref, không dị ứng thuốc nhưng dị ứng đậu phộng. ASR/MT làm mất thông tin dị ứng, thành không có thuốc nào trong vpin. Whisper cũng không nhận rõ input này; cần nghe/kiểm chất lượng nguồn.

- Input tham chiếu: not to any medication but i am allergic to peanuts
- ASR trong log: Not so any medication in vpin.
- Bản dịch: Không có loại thuốc nào trong vpin.
- Whisper nghe input: Not so in medication, but in the hemorrhagic penis.
- Whisper nghe output: không có loại thúc nào trong vợ tin.

### EV-5301 — cao / TN/TTS,MT

800 trong ngữ cảnh báo thức đã thành 08:00 đúng ở MT, nhưng heard là tám không; cần tám giờ. alarm nên là báo thức thay vì báo động. tts_err=0 không chứng minh chuẩn hóa giờ đúng.

- Input tham chiếu: just to confirm you set the alarm to go off at eight hundred
- ASR trong log: Just to confirm, you set the alarm to go off at 800.
- Bản dịch: Chỉ để xác nhận, anh đặt báo động lên 08:00.
- Whisper nghe input: Just to confirm, you set the alarm to go off at 800.
- Whisper nghe output: chỉ để xác nhận anh đặt báo động lên tắm không.

### EV-5455 — thấp / MT nhẹ

Giữ câu hỏi triệu chứng hiện còn hay đã hết. Câu lặp ngay bây giờ và dùng hoặc hơi cứng, chưa mất ý chính.

- Input tham chiếu: now do you have those these symptoms right now or have they gone off now
- ASR trong log: Now, do you have these symptoms right now or have they gone off?
- Bản dịch: Bây giờ, bạn có những triệu chứng này ngay bây giờ hoặc chúng đã biến mất?
- Whisper nghe input: Now, do you have those symptoms right now or have they gone off now?
- Whisper nghe output: Bây giờ bạn có những triệu chứng này ngay bây giờ hoặc chúng đã đến mất.

### EV-5482 — thấp / không có lỗi lớn ở mức text

Câu hỏi có nhớ lúc đó đang làm gì được dịch đúng và tự nhiên.

- Input tham chiếu: right okay do you remember what you were doing at the time
- ASR trong log: Right, okay, do you remember what you were doing at the time.
- Bản dịch: Được rồi, anh có nhớ lúc đó anh đang làm gì không?
- Whisper nghe input: Right, okay. Do you remember what you were doing at the time?
- Whisper nghe output: được rồi, anh có nhớ lúc đó anh đang làm gì không?

### EV-5600 — vừa / MT

Giữ ý dành thời gian với bạn bè và để xem sao; thời gian chất lượng... bên ngoài như là... là dịch từng chữ, nên diễn đạt tự nhiên hơn.

- Input tham chiếu: yeah i want some quality time quality time with my friends outside like it's it's really nice but we will see
- ASR trong log: Yeah, I want to some quality time with my friends and outside like it's really nice. But we will see.
- Bản dịch: Ừ, tôi muốn có thời gian chất lượng với bạn bè và bên ngoài như là nó thực sự tốt đẹp. Nhưng chúng ta sẽ thấy.
- Whisper nghe input: Yeah, I want to some quality time with my friends and outside like it's really nice. But we will see.
- Whisper nghe output: Cứ tôi muốn có thời gian chất lượng với bạn bè và bình ngoài như là nó thực sự tốt đẹp, nhưng chúng ta sẽ thấy."

### EV-5621 — cao / MT,TTS?

twelve noon/12 PM là 12 giờ trưa, MT đổi thành 12 giờ tối: sai thời điểm. Đây là lỗi nghĩa ở MT, không thể chỉ đổi cách đọc số ở TTS. Whisper output không rõ thêm phần giờ, cần nghe xác nhận phát âm.

- Input tham chiếu: tell me what was the temperature in our city at twelve noon
- ASR trong log: Tell me, what was the temperature in our city at 12 PM.
- Bản dịch: Nói cho tôi biết, nhiệt độ ở thành phố chúng tôi là gì lúc 12 giờ tối.
- Whisper nghe input: Tell me what was the temperature in our city at 12pm.
- Whisper nghe output: Nó cho tôi biết, nhiệt độ ở thành phố chúng tôi là gì lúc 10-12 giờ tôi".

### EV-5655 — thấp / không có lỗi lớn ở mức text

Bản dịch giữ đúng ý đăng trạng thái tận hưởng thời tiết đẹp. Tên riêng không chấm trong phạm vi này.

- Input tham chiếu: post a new status on facebook saying that i am enjoying the nice weather
- ASR trong log: Post a new status on Facebook saying that I am enjoying the nice weather.
- Bản dịch: Đăng một trạng thái mới trên Facebook nói rằng tôi đang tận hưởng thời tiết đẹp.
- Whisper nghe input: Post a new status on Facebook saying that I am enjoying the nice weather.
- Whisper nghe output: đang một trạng thái mới trên phầy sơn nói rằng tôi đang tận hưởng thời tiết đẹp.

### EV-5674 — thấp / MT nhẹ,TTS?

MT giữ ý xin bản tóm tắt tình hình. heard/Whisper khác nhau ở từ/tên cuối, không chấm tên riêng và chưa kết luận lỗi âm thanh tuyệt đối.

- Input tham chiếu: can i get a recap of what is going on with sprint right now
- ASR trong log: Can I get a recap of what is going on with sprint right now.
- Bản dịch: Tôi có thể có được một bản tóm tắt về những gì đang xảy ra với Sprint ngay bây giờ?
- Whisper nghe input: Can you get a recap of what is going on with the stretch right now?
- Whisper nghe output: tôi có thể có được một bản tóm tắt về những gì đang xảy ra với sách ngay bây giờ.
