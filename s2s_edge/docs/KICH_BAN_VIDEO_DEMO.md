# Kịch bản Quay Video Demo 30 Phút — AuraTranslate Edge (Phase 3)

- **Đội thi:** Gia Sư Đỉnh Cao
- **Dự án:** AuraTranslate Edge (OneVoice) — bản dựng `s2s_edge 0.1.0`
- **Thiết bị demo:** HSPTEK AIBOX 8550 (Qualcomm QCS8550, Hexagon NPU v73, RAM 16 GB LPDDR5x, Ubuntu 22.04 LTS, QAIRT 2.51)
- **Thời lượng mục tiêu:** 30 phút 00 giây
- **Căn cứ tiêu chí:** `docs/phase3_prototype_submission_brief.pdf` (Mục 3, 4, 6, 7.1) & `docs/phase3_prototype_submission_draft.md`

---

## I. Bố cục quay & Quy tắc bắt buộc từ Brief

### 1. Quy tắc bắt buộc không được vi phạm
- **Không cắt dựng / giả lập đoạn chạy demo (Mục 7.1):** Toàn bộ phần từ lúc **ngắt mạng** cho đến hết phần kiểm thử hội thoại và CLI phải quay **1 take liên tục**, hiển thị rõ đồng hồ hệ thống, giao diện web và terminal log để Ban Giám khảo đối chiếu thời gian xử lý thực tế.
- **Có bằng chứng ngắt mạng ngay trước khi chạy (Mục 7.1):** Quay rõ thao tác tắt Wi-Fi/LAN trên laptop và trên AIBOX 8550 (`nmcli networking off`), chạy `ping 8.8.8.8` báo lỗi không có kết nối.
- **Bám sát 4 trụ cột điểm (Mục 6):**
  - **A. Chất lượng dịch Voice-to-Voice (45%):** Trọn vẹn 6 chiều **VI ⇄ EN / KO / ZH** trong hội thoại tự nhiên tại phân xưởng sản xuất FDI.
  - **B. Hiệu năng On-device (20%):** Thời gian suy luận từng bước (`ASR / MT / TTS`), dung lượng RAM thực tế (`free -h`).
  - **C. Tích hợp thiết bị & UX (20%):** Giao diện `AuraTranslate Edge` trực quan (1 chạm / giữ `Space`), sửa chữ nhận dạng rồi dịch lại, kéo thả file âm thanh, hoạt động dưới tạp âm nền nhẹ.
  - **D. Hồ sơ demo (15%):** Quy trình tái lập rõ ràng, nêu trung thực giới hạn hiện tại và lộ trình phát triển.

### 2. Bố cục khung hình (OBS Studio) & Thu âm
- **Scene 1 — Toàn cảnh (dùng cho Intro `00:00–02:30` & Outro `28:15–30:00`):**
  - Camera góc rộng quay cả đội ngồi quanh bàn cùng **HSPTEK AIBOX 8550**, laptop điều khiển, loa ngoài và **micro USB** đặt ở trung tâm bàn.
- **Scene 2 — Picture-in-Picture (dùng xuyên suốt `02:30–28:15`):**
  - **80% khung hình chính (Screen Capture laptop):**
    - Bên trái (~60%): Trình duyệt mở **AuraTranslate Edge** (`http://localhost:8000`).
    - Bên phải (~40%): Terminal SSH/ADB vào AIBOX 8550 hiển thị log xử lý real-time + `free -h`.
  - **20% góc dưới bên phải (Camera PiP):** Quay cả đội đang ngồi thao tác trên laptop, nói vào micro USB và thấy hộp AIBOX 8550 đang chạy bên cạnh.
- **Âm thanh:** Thu qua **micro USB** đặt trên bàn (cân chỉnh trước để thu rõ cả giọng thành viên nói vào hệ thống lẫn giọng TTS phát ra từ loa).

---

## II. Phân vai & Checklist chuẩn bị trước khi bấm máy

### Phân vai
- **Vai A (Điều phối & Kỹ thuật):** Chạy lệnh terminal, thực hiện ngắt mạng, thao tác nút Mic / giữ `Space` / kéo thả `.wav` / sửa chữ (✎), giải thích thông số NPU và log.
- **Vai B (Quản lý FDI / Chuyên gia nước ngoài):** Đảm nhận thoại tiếng Anh, tiếng Hàn, tiếng Trung (có thể chia 2 người nếu đội có thành viên phụ trách riêng từng ngôn ngữ).
- **Vai C (Tổ trưởng / Công nhân / Thủ kho Việt Nam):** Đảm nhận thoại tiếng Việt, trực tiếp thao tác bấm Mic / `Space` trên giao diện web như người dùng tại xưởng.

### Checklist trước khi quay
- [ ] Cắm nguồn ổn định cho **HSPTEK AIBOX 8550**, cắm dây USB nối laptop (`adb devices` nhận hộp), cắm micro & loa USB vào đúng cổng.
- [ ] Chuẩn bị sẵn **1 chiếc quạt nhỏ** (đặt cạnh bàn, bật ở Hồi 4 để tạo tạp âm nền nhẹ 50–60 dB theo Mục 4 Brief).
- [ ] Chuẩn bị sẵn **1 file `.wav` mẫu** trên Desktop laptop để demo tính năng tải file / kéo thả ở Hồi 4.
- [ ] Quay thử 30 giây kiểm tra âm lượng micro USB thu cả tiếng người + tiếng loa TTS rõ, không rè, không bị quá nhỏ; chữ trên terminal đủ to để BGK đọc được trong video 1080p.
- [ ] In hoặc mở sẵn **Bảng câu thoại (Mục IV)** trên điện thoại/giấy để đọc tự nhiên, tránh ấp úng làm hụt mạch 1 take.

---

## III. Kịch bản chi tiết từng phút (00:00 – 30:00)

### Phần 1: Intro — Đội thi, Bài toán & Phần cứng (00:00 – 02:30 | 2 phút 30 giây)
*Khung hình: Scene 1 (Toàn đội bên thiết bị AIBOX 8550, laptop, mic USB).*

| Mốc | Hình ảnh / Thao tác | Nội dung trình bày |
|---|---|---|
| **00:00 – 00:45** | Cả đội chào trước camera. Đưa tay chỉ/cận cảnh nhẹ hộp **HSPTEK AIBOX 8550** và **micro USB** trên bàn. | **Vai A:** "Xin chào Ban Giám khảo OneVoice AI Challenge 2026. Chúng tôi là đội **Gia Sư Đỉnh Cao**, giới thiệu giải pháp **AuraTranslate Edge** — hệ thống phiên dịch giọng nói hai chiều chạy **100% offline trên thiết bị biên**." |
| **00:45 – 01:35** | Camera giữ góc toàn đội. | **Vai C:** "Bối cảnh giải pháp hướng tới là giao tiếp hằng ngày tại **nhà máy sản xuất FDI, công trường và kho vận** giữa quản lý, chuyên gia nước ngoài và công nhân Việt Nam — môi trường đòi hỏi bảo mật dữ liệu nội bộ, không phụ thuộc kết nối Internet và luôn có tạp âm nền từ máy móc, quạt gió." |
| **01:35 – 02:30** | Vai A giới thiệu phần cứng và bố cục buổi kiểm thử. | **Vai A:** "Toàn bộ 9 đồ thị AI của hệ thống được tối ưu và biên dịch chạy trên NPU Hexagon v73 của **HSPTEK AIBOX 8550** (Snapdragon QCS8550, RAM 16 GB), hỗ trợ trọn vẹn 6 chiều dịch **Việt ⇄ Anh / Trung / Hàn**. Ngay sau đây, đội sẽ ngắt toàn bộ mạng và thực hiện phiên hội thoại kiểm thử thực tế tại phân xưởng sản xuất trong gần 30 phút." |

---

### Phần 2: Kiểm tra NPU, Bằng chứng Ngắt mạng & Khởi động (02:30 – 06:00 | 3 phút 30 giây)
*Khung hình: Chuyển sang Scene 2 (PiP — 80% màn hình laptop + 20% camera cả đội). Giữ **1 take liên tục** từ đây tới 28:15.*

| Mốc | Thao tác trên màn hình & thiết bị | Lời dẫn |
|---|---|---|
| **02:30 – 03:30** | 1. Mở Terminal kết nối vào AIBOX (`adb shell` -> `su -`).<br>2. Chạy `bash start.sh check` (hoặc kiểm tra kết quả `tools/session0.sh` / `cat logs/models_ok.txt`) để hiển thị các đồ thị NPU đạt `PASS`, đồng thời gõ `arecord -l` cho thấy thiết bị audio USB đang kết nối với hộp. | **Vai A:** "Đầu tiên, chúng tôi kiểm tra nhanh trạng thái các đồ thị NPU trong gói offline và xác nhận micro/loa USB đã được hệ thống nhận diện." |
| **03:30 – 04:30** | **CẢNH BẮT BUỘC — NGẮT MẠNG:**<br>1. Trên Laptop: mở khay hệ thống Windows tắt **Wi-Fi** / bật **Airplane Mode**.<br>2. Trên Terminal AIBOX: chạy `nmcli networking off` (hoặc `ip link`), sau đó chạy `ping -c 3 8.8.8.8` ➔ màn hình báo `Network is unreachable`.<br>3. Trên Terminal phụ của Laptop: chạy `ping 8.8.8.8` ➔ thất bại hoàn toàn. | **Vai A:** "Để chứng minh hệ thống vận hành 100% offline, chúng tôi ngắt kết nối mạng trên cả laptop và AIBOX 8550. Lệnh `ping 8.8.8.8` trên cả hai thiết bị đều báo không thể kết nối. Từ lúc này toàn bộ suy luận ASR, dịch máy và tổng hợp giọng nói diễn ra hoàn toàn nội bộ trên chip QCS8550." |
| **04:30 – 06:00** | 1. Trên AIBOX: chạy `bash start.sh demo` (chờ hiện `OneVoice S2S demo -> http://127.0.0.1:8000`).<br>2. Trên Laptop: chạy `python s2s_edge_pkg/tools/laptop_web.py --demo` (mở port forward nội bộ qua cáp USB tới `http://localhost:8000`).<br>3. Trình duyệt hiện giao diện **AuraTranslate Edge** đã kết nối engine; cửa sổ terminal bên cạnh mở sẵn `free -h` cho thấy mức RAM thực tế (~2.2–3.5 GB / 16 GB). | **Vai A:** "Web Demo đã kết nối thành công qua `adb forward` nội bộ bằng cáp USB. BGK có thể quan sát giao diện AuraTranslate Edge bên trái và log suy luận cùng mức RAM tiêu thụ ở cửa sổ bên phải." |

---

### Phần 3: Hội thoại Thực tế Tại Phân xưởng Sản xuất FDI (06:00 – 26:00 | 20 phút)
*Khung hình: Scene 2 (PiP). Sau mỗi lượt nói, giữ yên 1–2 giây để loa phát hết câu dịch TTS và màn hình hiển thị đầy đủ thời gian `ASR / MT / TTS`.*

#### Hồi 1: Giao ban đầu ca & An toàn dây chuyền — Cặp VI ⇄ EN (06:00 – 11:30 | 5 phút 30 giây)
*Thao tác:* Click chip **`VI ↔ EN`**. Luân phiên nhấn đảo chiều hoặc chọn nguồn nói/đích, bấm Mic hoặc giữ phím `Space` để nói.

- **Lượt 1 (EN → VI | Vai B):** *"Good morning everyone. Before starting the morning shift, please check your anti-static wrist straps and safety goggles."*
- **Lượt 2 (VI → EN | Vai C):** *"Chào anh, toàn bộ mười hai công nhân tổ lắp ráp đã đeo vòng chống tĩnh điện và kiểm tra thiết bị bảo hộ đầy đủ rồi ạ."*
- **Lượt 3 (EN → VI | Vai B):** *"Great. Today's production target for line B2 is one thousand two hundred circuit boards, and the defect rate must stay below zero point five percent."*
- **Lượt 4 (VI → EN | Vai C):** *"Rõ rồi ạ. Hiện lô hàng mã A8 đang chạy ổn định, nhiệt độ lò hàn đang giữ ở mức hai trăm bốn mươi lăm độ C."*
- **Lượt 5 (EN → VI | Vai B):** *"Please record the serial numbers of batch A8 every two hours and report to the quality office before twelve o'clock."*
- **Lượt 6 (VI → EN | Vai C):** *"Vâng, cứ hai tiếng một lần tôi sẽ cập nhật số lượng đạt chuẩn và gửi báo cáo sang phòng kiểm tra chất lượng trước mười hai giờ trưa."*
- **Nhấn mạnh kỹ thuật (30 giây cuối Hồi 1 — Vai A):** Chỉ vào giao diện web/log chỉ rõ các số lượng, tỷ lệ phần trăm, nhiệt độ, giờ giấc được chuẩn hoá sang chữ để TTS đọc tự nhiên, cùng độ trễ từng khâu hiển thị rõ ràng.

#### Hồi 2: Xử lý sự cố kỹ thuật & Sửa nhận dạng trực tiếp — Cặp VI ⇄ KO (11:30 – 17:00 | 5 phút 30 giây)
*Thao tác:* Click chip **`VI ↔ KO`**. Lồng ghép tình huống bấm nút bút chì **✎** sửa chữ nhận dạng rồi dịch lại.

- **Lượt 1 (VI → KO | Vai C):** *"Anh quản lý ơi, máy gắp linh kiện số hai vừa báo lỗi lệch chân tụ điện trên bảng mạch."*
- **Lượt 2 (KO → VI | Vai B):** *"지금 바로 2번 라인을 잠깐 멈추고, AOI 검사기 화면을 같이 확인합시다."*
- **Lượt 3 (VI → KO | Vai C):** *"Em đã tạm dừng máy và kiểm tra đồng hồ áp suất khí nén, hiện đang ở mức không phẩy sáu megapascal."*
- **Lượt 4 (KO → VI | Vai B):** *"압력은 정상이네요. 흡착 노즐을 새 것으로 교체하고 샘플 스무 개만 다시 테스트해 주세요."*
- **Lượt 5 (Demo UX sửa chữ ✎ | Vai C & Vai A):**
  - **Vai C** nói một câu tiếng Việt có tên riêng hoặc từ dễ nhầm: *"Nhờ anh gọi kỹ thuật viên Minh sang kiểm tra lại đầu đọc mã vạch ở cổng số ba."*
  - Sau khi thẻ hội thoại hiện lên, **Vai A** click biểu tượng **bút chì ✎** trên thẻ đó, chỉnh sửa từ/chính tả rồi bấm **Enter** (hoặc nút **✓**) để hệ thống dịch và đọc lại ngay lập tức (thẻ hiển thị nhãn `"đã sửa tay"`).
  - **Vai A giải thích:** "Trong thực tế nhà máy có tên riêng hoặc mã thiết bị lạ, công nhân chỉ cần sửa nhanh 1–2 chữ trên màn hình mà không phải nói lại cả câu dài."
- **Lượt 6 (KO → VI | Vai B):** *"확인했습니다. 이상 없으면 10분 뒤에 라인을 다시 가동하세요."*

#### Hồi 3: Điều phối vật tư kho vận — Cặp VI ⇄ ZH (17:00 – 22:00 | 5 phút 00 giây)
*Thao tác:* Click chip **`VI ↔ ZH`**.

- **Lượt 1 (ZH → VI | Vai B):** *"你好，我们刚送到五十卷电阻和二十箱保护外壳，请问要放在哪个仓库？"*
- **Lượt 2 (VI → ZH | Vai C):** *"Chào anh, toàn bộ hàng mới nhập sẽ xếp vào khu B3, nhờ xe nâng chuyển giúp sang cửa kho số bốn trước mười bốn giờ ba mươi."*
- **Lượt 3 (ZH → VI | Vai B):** *"好的，但是这次还少了两箱塑料盖子，下午四点半的车会补送过来。"*
- **Lượt 4 (VI → ZH | Vai C):** *"Không sao, anh ghi chú mã phiếu nhập và để lại số điện thoại nội bộ để khi xe tới bảo vệ gọi trực tiếp cho tôi nhé."*
- **Lượt 5 (ZH → VI | Vai B):** *"没问题，我的分机号码是三零八五，货到了请随时联系我。"*

#### Hồi 4: Stress-test Môi trường Nhiễu Xưởng, Tải File & CLI (22:00 – 26:00 | 4 phút 00 giây)
*Mục tiêu: Đáp ứng Mục 4 (tạp âm nền nhẹ) & Mục 3 (nhiều giao diện cho BGK tự thao tác).*

1. **22:00 – 23:45 — Thử nghiệm dưới tiếng ồn quạt & khoảng cách 1.5 m:**
   - **Vai A** bật chiếc quạt nhỏ đặt gần bàn (nêu rõ cho BGK thấy trên camera PiP để tạo tạp âm nền tương đương môi trường kiểm thử).
   - **Vai C** ngồi lùi cách micro USB khoảng **1.5 m**, bấm giữ `Space` nói: *"Chuyền số một đã chạy lại ổn định, sản lượng đạt năm trăm sản phẩm đầu tiên."*
   - Đảo chiều thử thêm 1 câu **EN → VI** hoặc **KO → VI** trong khi quạt vẫn chạy: *"Please keep the aisle clear for the forklift."*
   - **Vai A:** Chỉ trên màn hình thấy bộ VAD thích ứng và lọc rác ASR vẫn bắt đúng câu nói, không bị tiếng quạt làm nhiễu hay sinh chữ ảo.
2. **23:45 – 24:45 — Kéo thả / Tải file `.wav` trên Web Demo:**
   - **Vai A** kéo thả (hoặc bấm nút Upload) 1 file `.wav` mẫu từ Desktop vào Web Demo.
   - Chỉ rõ thanh trạng thái `ASR → MT → TTS` chạy lần lượt trên giao diện và phát âm thanh kết quả.
3. **24:45 – 26:00 — Chạy chế độ CLI rảnh tay trên AIBOX:**
   - Tắt Web Demo bằng `Ctrl+C` trong terminal (giải thích việc giải phóng RAM giữa các chế độ trên bộ nhớ 16 GB).
   - Gõ lệnh chạy trực tiếp trên thiết bị:
     ```bash
     bash start.sh live vi ko
     ```
   - Nói nhanh 1 lượt **VI → KO** và 1 lượt **KO → VI** qua mic/loa USB gắn trực tiếp với hộp để chứng minh ngoài Web UI, giám khảo có thể dùng trực tiếp CLI rảnh tay.

---

### Phần 4: Tổng kết Hiệu năng, Giới hạn & Outro (26:00 – 30:00 | 4 phút 00 giây)

#### 4.1. Đọc số liệu thực tế, Giới hạn & Lộ trình (26:00 – 28:15 | 2 phút 15 giây)
*Khung hình: Scene 2 (PiP) — Vai A mở cửa sổ terminal/log và `free -h` vừa đo được.*

- **26:00 – 27:00 — Số liệu thực đo trên thiết bị (Tiêu chí B):**
  - Trỏ vào terminal log ghi lại suốt phiên chạy vừa qua (thời gian `ASR`, `MT`, `TTS`, tổng thời gian từng lượt và RAM đang chiếm dụng).
  - Nhấn mạnh: cả 9 đồ thị context binary (~4.0 GB lưu trữ, ~2.2 GB RAM khi nạp một chiều dịch) chạy hoàn toàn trên NPU Hexagon v73, ổn định liên tục suốt hơn 20 phút kiểm thử mà không rò rỉ bộ nhớ hay crash.
- **27:00 – 28:15 — Giới hạn hiện tại & Hướng phát triển (Mục 3 Brief):**
  - **Giới hạn hiện tại:** (1) Tên riêng ngoại lai hoặc mã kỹ thuật ít gặp đôi khi cần sửa tay trên Web UI; (2) Nói chồng âm nhiều người cùng lúc sát micro dễ lẫn tiếng; (3) Câu pha trộn nhiều từ mượn tiếng Anh trong tiếng Việt khi ồn cao.
  - **Lộ trình tiếp theo:** (1) Dùng Greedy Search cho câu ngắn dưới 12 token để giảm ~50% độ trễ MT; (2) Pipelining chồng chập (chạy ASR câu sau song song lúc đang phát TTS câu trước); (3) Bổ sung từ điển phát âm G2P chuyên biệt cho thuật ngữ nhà máy.

#### 4.2. Outro — Kết thúc & Cảm ơn (28:15 – 30:00 | 1 phút 45 giây)
*Khung hình: Chuyển về Scene 1 (Cả đội ngồi quanh thiết bị AIBOX 8550, laptop, mic USB).*

- **28:15 – 29:15:** Đại diện đội tổng kết 3 điểm cốt lõi của **AuraTranslate Edge**:
  1. **100% On-device & Offline** trên phần cứng thực tế HSPTEK AIBOX 8550.
  2. **Đa ngôn ngữ trọn vẹn 6 chiều (VI ⇄ EN / ZH / KO)** phù hợp môi trường nhà máy FDI, kho vận.
  3. **UX thực tế cho người lao động:** thao tác đơn giản, hỗ trợ sửa chữ dịch lại, chạy ổn định khi có tạp âm nền nhẹ.
- **29:15 – 30:00:** Cả đội gửi lời cảm ơn tới Ban Tổ chức **OneVoice AI Challenge 2026**, Qualcomm và Ban Giám khảo. *(Giữ 5–10 giây cuối hiển thị tên đội **Gia Sư Đỉnh Cao**, dự án **AuraTranslate Edge**, bản build `s2s_edge 0.1.0`).*

---

## IV. Bảng câu thoại nhanh (Cue Sheet để in / mở khi quay)

| Hồi | Chiều | Người nói | Câu thoại chuẩn |
|---|---|---|---|
| **Hồi 1** | EN → VI | Vai B | Good morning everyone. Before starting the morning shift, please check your anti-static wrist straps and safety goggles. |
| **Hồi 1** | VI → EN | Vai C | Chào anh, toàn bộ mười hai công nhân tổ lắp ráp đã đeo vòng chống tĩnh điện và kiểm tra thiết bị bảo hộ đầy đủ rồi ạ. |
| **Hồi 1** | EN → VI | Vai B | Great. Today's production target for line B2 is one thousand two hundred circuit boards, and the defect rate must stay below zero point five percent. |
| **Hồi 1** | VI → EN | Vai C | Rõ rồi ạ. Hiện lô hàng mã A8 đang chạy ổn định, nhiệt độ lò hàn đang giữ ở mức hai trăm bốn mươi lăm độ C. |
| **Hồi 1** | EN → VI | Vai B | Please record the serial numbers of batch A8 every two hours and report to the quality office before twelve o'clock. |
| **Hồi 1** | VI → EN | Vai C | Vâng, cứ hai tiếng một lần tôi sẽ cập nhật số lượng đạt chuẩn và gửi báo cáo sang phòng kiểm tra chất lượng trước mười hai giờ trưa. |
| **Hồi 2** | VI → KO | Vai C | Anh quản lý ơi, máy gắp linh kiện số hai vừa báo lỗi lệch chân tụ điện trên bảng mạch. |
| **Hồi 2** | KO → VI | Vai B | 지금 바로 2번 라인을 잠깐 멈추고, AOI 검사기 화면을 같이 확인합시다. |
| **Hồi 2** | VI → KO | Vai C | Em đã tạm dừng máy và kiểm tra đồng hồ áp suất khí nén, hiện đang ở mức không phẩy sáu megapascal. |
| **Hồi 2** | KO → VI | Vai B | 압력은 정상이네요. 흡착 노즐을 새 것으로 교체하고 샘플 스무 개만 다시 테스트해 주세요. |
| **Hồi 2** | VI → KO (✎) | Vai C | Nhờ anh gọi kỹ thuật viên Minh sang kiểm tra lại đầu đọc mã vạch ở cổng số ba. |
| **Hồi 2** | KO → VI | Vai B | 확인했습니다. 이상 없으면 10분 뒤에 라인을 다시 가동하세요. |
| **Hồi 3** | ZH → VI | Vai B | 你好，我们刚送到五十卷电阻和二十箱保护外壳，请问要放在哪个仓库？ |
| **Hồi 3** | VI → ZH | Vai C | Chào anh, toàn bộ hàng mới nhập sẽ xếp vào khu B3, nhờ xe nâng chuyển giúp sang cửa kho số bốn trước mười bốn giờ ba mươi. |
| **Hồi 3** | ZH → VI | Vai B | 好的，但是这次还少了两箱塑料盖子，下午四点半的车会补送过来。 |
| **Hồi 3** | VI → ZH | Vai C | Không sao, anh ghi chú mã phiếu nhập và để lại số điện thoại nội bộ để khi xe tới bảo vệ gọi trực tiếp cho tôi nhé. |
| **Hồi 3** | ZH → VI | Vai B | 没问题，我的分机号码是三零八五，货到了请随时联系我。 |
| **Hồi 4** | VI → EN (1.5m + quạt) | Vai C | Chuyền số một đã chạy lại ổn định, sản lượng đạt năm trăm sản phẩm đầu tiên. |
| **Hồi 4** | EN → VI (1.5m + quạt) | Vai B | Please keep the aisle clear for the forklift. |
