# Kịch Bản Quay Video Demo 30 Phút (Tuần Tự Từng Bước)

- **Đội thi:** Gia Sư Đỉnh Cao — **Dự án:** AuraTranslate Edge (OneVoice)
- **Thiết bị trên bàn khi quay:**
  1. Hộp **HSPTEK AIBOX 8550** (chạy NPU Hexagon v73, nối cáp USB sang laptop).
  2. **Micro USB RGB ME6S** đặt giữa bàn (dùng để cả nhóm nói trực tiếp khi kiểm thử).
  3. Laptop **Lenovo LOQ** (hiển thị màn hình điều khiển, đồng thời dùng **loa của laptop Lenovo LOQ** phát âm thanh dịch đầu ra).
  4. **Điện thoại / Máy quay** (để quay toàn cảnh cả nhóm và lia vào màn hình laptop) + **Phần mềm quay màn hình (OBS)** trên laptop (để lấy hình màn hình rõ nét).

> **Lưu ý quan trọng từ BTC:** Từ lúc tắt mạng (Bước 2) cho đến hết phần chạy test (Bước 4), giữ máy quay và trình quay màn hình chạy **liên tục không cắt ghép** để Ban Giám khảo thấy đúng thời gian xử lý thực tế.

---

## BƯỚC 1: INTRO — Mới vô quay toàn cảnh cả nhóm & giới thiệu thiết bị (00:00 – 02:00)

1. **Mới bắt đầu bấm máy:**
   - Đặt góc máy quay **toàn cảnh cả nhóm** đang ngồi quanh bàn làm việc. Trên bàn đặt sẵn hộp **HSPTEK AIBOX 8550**, chiếc **Micro USB RGB ME6S** bật đèn sẵn, và chiếc laptop **Lenovo LOQ** đang mở.
2. **Cả nhóm chào Ban Giám khảo & giới thiệu:**
   - **Đại diện nhóm nói:** *"Xin chào Ban Giám khảo OneVoice AI Challenge 2026. Chúng tôi là đội Gia Sư Đỉnh Cao, mang đến giải pháp AuraTranslate Edge — thiết bị dịch thuật giọng nói đa ngôn ngữ chạy hoàn toàn offline trên phần cứng biên, phục vụ giao tiếp giữa quản lý nước ngoài và công nhân Việt Nam tại các nhà máy sản xuất FDI và kho vận."*
3. **Lia camera giới thiệu từng thiết bị trên bàn:**
   - Người cầm máy **lia camera cận cảnh** vào hộp **HSPTEK AIBOX 8550** trên bàn -> thành viên chỉ vào hộp và giới thiệu: *"Đây là thiết bị xử lý chính HSPTEK AIBOX 8550 sử dụng chip Qualcomm QCS8550 với NPU Hexagon v73 và 16 GB RAM, nơi toàn bộ 9 đồ thị AI nhận dạng, dịch máy và tổng hợp giọng nói được thực thi."*
   - Tiếp tục **lia camera sang chiếc Micro USB RGB ME6S và laptop Lenovo LOQ** -> giới thiệu: *"Để thu âm kiểm thử thực tế hôm nay, nhóm sử dụng Micro USB định hướng ME6S đặt trên bàn, kết nối với giao diện điều khiển trên laptop Lenovo LOQ qua cáp USB nội bộ, và phát âm thanh dịch trực tiếp ra loa của laptop để Ban Giám khảo cùng nghe rõ."*
   - Chốt Intro: *"Bây giờ nhóm sẽ tiến hành ngắt toàn bộ kết nối mạng, kiểm thử hội thoại trực tiếp qua micro trên cả 6 chiều dịch Việt – Anh – Hàn – Trung, và chạy kiểm thử trực tiếp trên bộ dữ liệu dataset mà nhóm đã chuẩn bị."*

---

## BƯỚC 2: Ngắt mạng & Kiểm tra NPU sẵn sàng (02:00 – 05:00)

1. **Lia camera vào màn hình laptop Lenovo LOQ:**
   - Người cầm máy di chuyển góc quay **chiếu thẳng vào màn hình laptop** (đồng thời trên laptop đang ghi màn hình).
2. **Thao tác ngắt mạng (Bắt buộc có trong video):**
   - Thành viên ngồi máy bấm vào góc dưới bên phải màn hình Windows trên laptop, **tắt Wi-Fi / bật chế độ Máy bay (Airplane Mode)**.
   - Mở cửa sổ Terminal đang kết nối với hộp AIBOX (`adb shell`), gõ lệnh tắt mạng trên hộp:
     ```bash
     nmcli networking off
     ping -c 3 8.8.8.8
     ```
   - Chỉ tay vào màn hình cho camera thấy rõ dòng chữ `Network is unreachable` (và trên laptop gõ `ping 8.8.8.8` cũng báo lỗi mất mạng).
   - **Thuyết minh ngắn:** *"Như Ban Giám khảo thấy, cả hộp AIBOX 8550 và laptop đều đã ngắt mạng hoàn toàn. Mọi xử lý từ giây phút này đều diễn ra 100% offline trên thiết bị."*
3. **Kiểm tra trạng thái NPU & RAM trước khi chạy:**
   - Trên Terminal của hộp, chạy nhanh kiểm tra model và bộ nhớ:
     ```bash
     cat logs/models_ok.txt
     free -h
     ```
   - Chỉ vào màn hình cho thấy các đồ thị NPU đã kiểm tra đạt chuẩn (`PASS`) và lượng RAM trống còn rất dồi dào trên tổng 16 GB.

---

## BƯỚC 3: Cả nhóm thu âm kiểm thử trực tiếp qua Mic USB & Nghe loa Laptop phát Output (05:00 – 21:00)

### Khởi động giao diện Web Demo (05:00 – 06:00)
- Trên cửa sổ Terminal của hộp AIBOX, chạy:
  ```bash
  bash start.sh demo
  ```
- Trên cửa sổ Terminal của laptop, chạy:
  ```bash
  python s2s_edge_pkg/tools/laptop_web.py --demo
  ```
- Trình duyệt trên laptop mở giao diện **AuraTranslate Edge** (`http://localhost:8000`). Để cửa sổ trình duyệt chiếm 2/3 màn hình bên trái, cửa sổ Terminal log của hộp nằm 1/3 bên phải để thấy log chạy thật.

> **Cách phối hợp máy quay trong suốt Bước 3:**
> - **Khi thành viên nói:** Giữ góc máy **toàn cảnh hoặc trung cảnh thấy thành viên trong nhóm đang nói trực tiếp vào chiếc Micro USB ME6S** và tay bấm nút Mic (hoặc giữ phím `Space`) trên laptop.
> - **Ngay khi vừa dứt câu nói:** **Lia camera sát vào màn hình laptop** để thấy rõ thanh tiến độ `ASR → MT → TTS` chạy, chữ nhận dạng và bản dịch hiện ra, đồng thời cả nhóm giữ yên 2–3 giây để **loa laptop Lenovo LOQ phát to câu dịch ra ngoài**.

---

### 3.1. Kiểm thử Cặp Tiếng Việt ⇄ Tiếng Anh (`VI ↔ EN`) (06:00 – 11:00)
*Bối cảnh: Quản lý nhà máy người nước ngoài và Tổ trưởng sản xuất Việt Nam trao đổi đầu ca sáng.*

1. **Thao tác:** Bấm vào nút chọn nhanh **`VI ↔ EN`** trên màn hình.
2. **Lượt 1 (EN → VI):**
   - Quay cảnh **Thành viên B** nói vào **Mic ME6S**:
     > *"Good morning everyone. Before starting the morning shift, please check your anti-static wrist straps and safety goggles."*
   - Lia camera vào màn hình laptop: xem chữ tiếng Anh hiện lên, bản dịch tiếng Việt hiện ra kèm thời gian xử lý (ms), và **nghe loa laptop Lenovo LOQ đọc câu tiếng Việt**.
3. **Lượt 2 (VI → EN):**
   - Bấm nút đảo chiều `⇄` (hoặc chọn nguồn Tiếng Việt). Quay cảnh **Thành viên C** nói vào **Mic ME6S**:
     > *"Chào anh, toàn bộ mười hai công nhân tổ lắp ráp đã đeo vòng chống tĩnh điện và kiểm tra thiết bị bảo hộ đầy đủ rồi ạ."*
   - Lia camera vào màn hình xem câu nhận dạng có tự động thêm dấu câu, bản dịch tiếng Anh hiện ra và loa laptop phát giọng đọc tiếng Anh.
4. **Lượt 3 (EN → VI):**
   - Quay **Thành viên B** nói vào mic:
     > *"Great. Today's production target for line B2 is one thousand two hundred circuit boards, and the defect rate must stay below zero point five percent."*
   - Lia vào màn hình xem kết quả và nghe loa laptop phát (chú ý các con số `1.200`, `0.5%` được đọc thành chữ rõ ràng).
5. **Lượt 4 (VI → EN):**
   - Quay **Thành viên C** nói vào mic:
     > *"Rõ rồi ạ. Hiện lô hàng mã A8 đang chạy ổn định, nhiệt độ lò hàn đang giữ ở mức hai trăm bốn mươi lăm độ C."*
   - Lia vào màn hình xem kết quả và nghe loa phát.
6. **Lượt 5 (EN → VI):**
   - Quay **Thành viên B** nói vào mic:
     > *"Please record the serial numbers of batch A8 every two hours and report to the quality office before twelve o'clock."*
   - Lia vào màn hình và nghe loa phát.
7. **Lượt 6 (VI → EN):**
   - Quay **Thành viên C** nói vào mic:
     > *"Vâng, cứ hai tiếng một lần tôi sẽ cập nhật số lượng đạt chuẩn và gửi báo cáo sang phòng kiểm tra chất lượng trước mười hai giờ trưa."*
   - Lia vào màn hình xem thời gian xử lý từng khâu `ASR / MT / TTS` và nghe loa phát xong.

---

### 3.2. Kiểm thử Cặp Tiếng Việt ⇄ Tiếng Hàn (`VI ↔ KO`) & Demo sửa chữ nhận dạng (11:00 – 16:00)
*Bối cảnh: Quản lý Hàn Quốc và công nhân Việt Nam xử lý lỗi lệch chân linh kiện trên chuyền SMT.*

1. **Thao tác:** Bấm vào nút **`VI ↔ KO`** trên màn hình.
2. **Lượt 1 (VI → KO):**
   - Quay cảnh **Thành viên C** nói vào **Mic ME6S**:
     > *"Anh quản lý ơi, máy gắp linh kiện số hai vừa báo lỗi lệch chân tụ điện trên bảng mạch."*
   - Lia camera vào màn hình xem bản dịch tiếng Hàn hiện ra và nghe loa laptop đọc câu tiếng Hàn.
3. **Lượt 2 (KO → VI):**
   - Đảo chiều sang `KO → VI`. Quay cảnh **Thành viên B** nói vào mic (hoặc phát/đọc mẫu câu tiếng Hàn chuẩn):
     > *"지금 바로 2번 라인을 잠깐 멈추고, AOI 검사기 화면을 같이 확인합시다."*
     *(Nghĩa: Hãy dừng nhanh chuyền số 2 ngay bây giờ và cùng kiểm tra màn hình máy kiểm tra AOI.)*
   - Lia camera vào màn hình xem bản dịch tiếng Việt và nghe loa laptop đọc.
4. **Lượt 3 (VI → KO):**
   - Quay **Thành viên C** nói vào mic:
     > *"Em đã tạm dừng máy và kiểm tra đồng hồ áp suất khí nén, hiện đang ở mức không phẩy sáu megapascal."*
   - Lia camera vào màn hình và nghe loa laptop phát.
5. **Lượt 4 (KO → VI):**
   - Quay **Thành viên B** nói vào mic:
     > *"압력은 정상이네요. 흡착 노즐을 새 것으로 교체하고 샘플 스무 개만 다시 테스트해 주세요."*
     *(Nghĩa: Áp suất bình thường đấy. Hãy thay đầu hút mới và chạy thử lại 20 sản phẩm mẫu nhé.)*
   - Lia camera vào màn hình và nghe loa phát.
6. **Lượt 5 — Demo tính năng sửa nhanh chữ nhận dạng rồi dịch lại (Điểm ăn điểm UX):**
   - Quay **Thành viên C** nói vào mic một câu có tên riêng:
     > *"Nhờ anh gọi kỹ thuật viên Minh sang kiểm tra lại đầu đọc mã vạch ở cổng số ba."*
   - Lia camera sát vào màn hình laptop: Thành viên điều khiển chỉ tay vào nút **hình bút chì (✎)** ngay trên thẻ hội thoại vừa hiện ra, bấm vào đó, sửa nhanh 1 từ (ví dụ sửa tên riêng hoặc mã cổng) rồi bấm **Enter / dấu tích (✓)**.
   - Máy lập tức dịch lại và **loa laptop Lenovo LOQ đọc lại bản dịch mới**, trên thẻ hiện chữ `"đã sửa tay"`.
   - **Thuyết minh ngắn:** *"Trong môi trường xưởng ồn, nếu nhận dạng nhầm một tên riêng hay mã máy, công nhân chỉ cần bấm nút bút chì sửa lại chữ đó rồi bấm Enter là máy dịch và đọc lại ngay, không cần phải nói lại cả câu."*
7. **Lượt 6 (KO → VI):**
   - Quay **Thành viên B** nói chốt:
     > *"확인했습니다. 이상 없으면 10분 뒤에 라인을 다시 가동하세요."*
     *(Nghĩa: Tôi đã rõ. Nếu không có bất thường nào thì 10 phút nữa cho chuyền chạy lại nhé.)*

---

### 3.3. Kiểm thử Cặp Tiếng Việt ⇄ Tiếng Trung (`VI ↔ ZH`) (16:00 – 19:30)
*Bối cảnh: Thủ kho Việt Nam và Giám sát kho nói tiếng Trung đối chiếu đơn nhập hàng.*

1. **Thao tác:** Bấm vào nút **`VI ↔ ZH`** trên màn hình.
2. **Lượt 1 (ZH → VI):**
   - Quay **Thành viên B** nói vào **Mic ME6S**:
     > *"你好，我们刚送到五十卷电阻和二十箱保护外壳，请问要放在哪个仓库？"*
     *(Nghĩa: Xin chào, chúng tôi vừa giao 50 cuộn điện trở và 20 thùng vỏ bảo vệ, xin hỏi cần để ở kho nào?)*
   - Lia camera vào màn hình xem bản dịch tiếng Việt và nghe loa laptop đọc.
3. **Lượt 2 (VI → ZH):**
   - Đảo chiều `VI → ZH`. Quay **Thành viên C** nói vào mic:
     > *"Chào anh, toàn bộ hàng mới nhập sẽ xếp vào khu B3, nhờ xe nâng chuyển giúp sang cửa kho số bốn trước mười bốn giờ ba mươi."*
   - Lia camera vào màn hình xem chữ Hán hiện ra và nghe **loa laptop phát giọng đọc MeloTTS tiếng Trung**.
4. **Lượt 3 (ZH → VI):**
   - Quay **Thành viên B** nói vào mic:
     > *"好的，但是这次还少了两箱塑料盖子，下午四点半的车会补送过来。"*
     *(Nghĩa: Được rồi, nhưng lần này còn thiếu 2 thùng nắp nhựa, chuyến xe lúc 4 rưỡi chiều sẽ giao bù sang.)*
   - Lia camera vào màn hình và nghe loa laptop phát.
5. **Lượt 4 (VI → ZH):**
   - Quay **Thành viên C** nói vào mic:
     > *"Không sao, anh ghi chú mã phiếu nhập và để lại số điện thoại nội bộ để khi xe tới bảo vệ gọi trực tiếp cho tôi nhé."*
   - Lia camera vào màn hình và nghe loa laptop phát.
6. **Lượt 5 (ZH → VI):**
   - Quay **Thành viên B** nói vào mic:
     > *"没问题，我的分机号码是三零八五，货到了请随时联系我。"*
     *(Nghĩa: Không vấn đề gì, số máy lẻ của tôi là 3085, hàng tới cứ liên hệ tôi bất cứ lúc nào.)*
   - Lia camera vào màn hình và nghe loa laptop phát.

---

### 3.4. Thử nghiệm khoảng cách xa + tạp âm quạt & Tải file `.wav` (19:30 – 21:00)

1. **Thử nói khi có tạp âm quạt và ngồi cách mic 1,5m:**
   - Bật quạt gió đặt gần bàn làm việc. Một thành viên ngồi lùi ra cách **Micro USB ME6S** khoảng **1m – 1,5m**, người ngồi máy bấm giữ phím `Space` để thu:
     > *"Chuyền số một đã chạy lại ổn định, sản lượng đạt năm trăm sản phẩm đầu tiên."*
   - Lia camera vào màn hình laptop cho thấy dù có tiếng quạt và ngồi xa, bộ lọc VAD + làm sạch ASR vẫn nhận đúng câu và loa laptop phát bản dịch rõ ràng.
2. **Thử kéo thả / tải 1 file `.wav` vào Web Demo:**
   - Trên laptop, bấm nút **Tải file âm thanh** (hoặc kéo thả 1 file `.wav` mẫu từ màn hình Desktop thả vào giao diện **AuraTranslate Edge**).
   - Lia camera vào màn hình để thấy máy giải mã file `.wav`, chạy qua `ASR → MT → TTS` và phát tiếng ra loa laptop.

---

## BƯỚC 4: Quay màn hình Laptop chạy Test một phần Bộ Dataset chuẩn bị sẵn (21:00 – 27:30)

*Mục tiêu: Chứng minh cho Ban Giám khảo thấy hệ thống chạy thực tế và chấm điểm tự động trên bộ dataset chuẩn bị sẵn (`BTC` / `ADV`), nhưng **chỉ chọn chạy 1 phần nhỏ (ví dụ 5–10 mẫu mỗi chiều)** để vừa khít thời lượng video mà không phải đợi hàng trăm câu.*

1. **Chuyển từ Web Demo sang công cụ chạy Dataset (21:00 – 22:00):**
   - Lia camera vào màn hình laptop (kết hợp quay màn hình OBS).
   - Trên cửa sổ Terminal của hộp AIBOX, nhấn `Ctrl+C` để đóng Web Demo (giải thích ngắn: *"Nhóm tắt Web Demo để giải phóng RAM trên hộp trước khi nạp trình chạy đánh giá bộ dữ liệu"*).
   - Khởi động trình chạy kiểm thử bộ dữ liệu trên hộp:
     ```bash
     bash start.sh box
     ```
     *(Hoặc `bash tools/box_serve.sh`)*.
   - Trên cửa sổ Terminal của laptop, chạy:
     ```bash
     python s2s_edge_pkg/tools/laptop_web.py
     ```
     -> Trình duyệt mở trang điều khiển & đánh giá dữ liệu tại `http://127.0.0.1:8770`.
     *(Lưu ý: Nếu nhóm muốn chạy trực tiếp bằng dòng lệnh trên Terminal thay vì giao diện web 8770, có thể chạy lệnh `bash tools/vng_run_all.sh /tmp/onevoice 5` để chạy tự động 5 mẫu/chiều ngay trên Terminal).*

2. **Chọn chạy 1 phần Dataset trên màn hình (22:00 – 26:00):**
   - Trên giao diện **Tab "Chạy bộ dữ liệu"** (hoặc trên cửa sổ Terminal):
     - Chọn bộ dữ liệu **BTC** (bộ dữ liệu có câu chuẩn và bản dịch tham chiếu).
     - Chọn chiều dịch kiểm thử (ví dụ `vi-en`, `en-vi`, `vi-ko`, `ko-vi`...).
     - Ở ô **Số clip mỗi chiều**, nhập **`5`** (hoặc **`8`** câu) — giải thích rõ với BGK: *"Vì bộ dữ liệu đầy đủ gồm từ 100 đến 180 câu mỗi chiều mất nhiều thời gian, trong video này nhóm chạy trực tiếp một phần gồm 5–8 mẫu đại diện ngay trên NPU để Ban Giám khảo quan sát toàn bộ quá trình chấm điểm thời gian thực; kết quả chạy toàn bộ bộ dữ liệu đã được nhóm xuất đầy đủ trong hồ sơ nộp bài."*
     - Bấm nút **"Chạy trên chip"**.
   - **Giữ cảnh quay màn hình laptop (xen kẽ góc camera lia vào màn hình và góc cả nhóm đang theo dõi):**
     - Quan sát từng dòng mẫu dữ liệu hiện lên lần lượt khi NPU xử lý xong:
       - Cột **Âm thanh gốc & Câu chuẩn**.
       - Cột **Kết quả ASR**, **thời gian chạy ASR (ms)** và **tỷ lệ lỗi CER/WER**.
       - Cột **Kết quả dịch MT**, **điểm chất lượng chrF**, và **thời gian chạy MT (ms)**.
       - Cột **Âm thanh đầu ra TTS**, **thời gian chạy TTS (ms)** và **tổng thời gian chạy trên NPU**.
     - Khi chạy xong 5–8 mẫu, thành viên ngồi máy **bấm nút Play (▶) ở 2–3 dòng bất kỳ** để phát cả **âm thanh gốc trong dataset** lẫn **âm thanh dịch do AIBOX 8550 tạo ra** qua **loa laptop Lenovo LOQ** cho Ban Giám khảo nghe đối chứng trực tiếp.

3. **Thử thêm 2–3 mẫu của Bộ dữ liệu Nhiễu (`ADV`) & Xem bảng tổng hợp (26:00 – 27:30):**
   - Chuyển sang chọn bộ **ADV (bộ dữ liệu nhiễu nhà xưởng / công trường 15 dB – 10 dB)**, đặt số lượng **3 câu**, bấm **"Chạy trên chip"**.
   - Sau khi chạy xong, cuộn lên **Bảng tổng hợp kết quả** trên màn hình: chỉ vào các con số trung bình (lỗi ASR, điểm chrF, thời gian trung bình từng chặng `ASR / MT / TTS`, phân vị `p50 / p90`, RTF) và bấm nút **"Tải CSV"** để lưu file kết quả.

---

## BƯỚC 5: OUTRO — Quay toàn cảnh cả nhóm tổng kết & Kết thúc (27:30 – 30:00)

1. **Lia camera từ màn hình laptop ra lại toàn cảnh cả nhóm (27:30 – 28:30):**
   - Người cầm máy lùi ra để lấy lại khung hình **toàn cảnh cả nhóm Gia Sư Đỉnh Cao** ngồi quanh bàn cùng hộp **HSPTEK AIBOX 8550**, **Micro USB RGB ME6S** và laptop **Lenovo LOQ**.
   - **Thành viên đại diện tổng kết kết quả vừa chạy:**
     > *"Qua gần 30 phút kiểm thử liên tục không ngắt quãng trong điều kiện tắt mạng hoàn toàn, hệ thống AuraTranslate Edge trên hộp HSPTEK AIBOX 8550 đã hoàn thành cả hai phần: thu âm hội thoại trực tiếp qua Micro USB ME6S phát ra loa laptop trên trọn vẹn 6 chiều dịch Việt – Anh – Hàn – Trung, và chạy chấm điểm trực tiếp trên bộ dữ liệu mẫu với đầy đủ thông số độ trễ và độ chính xác trên NPU Hexagon v73."*

2. **Nêu ngắn gọn giới hạn hiện tại & hướng phát triển tiếp theo (28:30 – 29:30):**
   - **Thành viên kỹ thuật chia sẻ thêm (đúng yêu cầu khuyến khích của BTC):**
     > *"Về giới hạn hiện tại của bản nguyên mẫu: thứ nhất, với một số tên riêng nước ngoài hoặc mã viết tắt đặc thù chưa có trong từ điển, đôi khi người dùng cần bấm nút sửa nhanh trên giao diện web như nhóm vừa minh hoạ; thứ hai, trường hợp nhiều người nói đè lên nhau cùng lúc sát micro sẽ làm giảm độ chính xác nhận dạng. Trong giai đoạn tiếp theo hướng tới vòng Field-Testing và Chung kết, nhóm sẽ kích hoạt chế độ giải mã nhanh cho các câu ngắn dưới 12 từ để giảm thêm một nửa độ trễ dịch, chạy gối đầu nhận dạng câu sau trong lúc đang phát loa câu trước, và bổ sung từ điển phát âm chuyên ngành nhà máy."*

3. **Lời chào kết thúc (29:30 – 30:00):**
   - **Cả nhóm cùng nhìn vào camera:**
     > *"Đội Gia Sư Đỉnh Cao xin chân thành cảm ơn Ban Tổ chức OneVoice AI Challenge 2026, Qualcomm và Ban Giám khảo đã theo dõi phần trình diễn nguyên mẫu AuraTranslate Edge. Xin cảm ơn và hẹn gặp lại tại vòng Field-Testing!"*
   - Cả nhóm vẫy tay chào, giữ khung hình ổn định 3–5 giây rồi bấm dừng quay.
